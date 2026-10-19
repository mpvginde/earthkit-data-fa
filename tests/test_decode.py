import os
import shutil

import numpy as np
import pytest
from conftest import sample

from earthkit_data_fa.decode import _unpack_bits, decode
from earthkit_data_fa.fafile import FAFile
from earthkit_data_fa.geography import FAGeography


def read(f, name):
    record = next(r for r in f.records if r.name == name)
    return decode(f.read(record), f.frame)


@pytest.mark.parametrize("nbits", range(1, 33))
def test_unpack_bits(nbits):
    n = 1001
    values = np.random.default_rng(nbits).integers(0, 2**nbits, n, dtype=np.uint64)
    bits = "".join(format(int(v), f"0{nbits}b") for v in values)
    bits += "0" * (-len(bits) % 8)
    raw = b"xx" + int(bits, 2).to_bytes(len(bits) // 8, "big")

    assert np.array_equal(_unpack_bits(raw, 2, nbits, n), values.astype(np.float64))


def test_spectral_matches_gridpoint_on_non_square_domain():
    f = FAFile(sample("clim"))
    assert f.frame.ndlon != f.frame.ndgl

    spectral = read(f, "SPECSURFGEOPOTEN")
    gridpoint = read(f, "SURFGEOPOTENTIEL")

    assert spectral.shape == gridpoint.shape == (69, 49)
    assert np.abs(spectral - gridpoint).max() < 1e-9


@pytest.mark.parametrize("name", ["lambert_HN", "mercator_HS", "stereopol_HN"])
def test_spectral_matches_gridpoint_on_trimmed_domain(name):
    # 320x320 stored points, 300x300 without the extension zone
    f = FAFile(sample(name))

    spectral = read(f, "SPECSURFGEOPOTEN")
    gridpoint = read(f, "SURFGEOPOTENTIEL")

    assert spectral.shape == gridpoint.shape == (300, 300)
    assert np.abs(spectral - gridpoint).max() < 1e-8 * np.abs(gridpoint).max()
    assert set(np.unique(read(f, "SURFIND.TERREMER"))) <= {0.0, 1.0}


def test_grib2_records():
    f = FAFile(sample("clim"))
    assert sum(f.header(r).ngrib >= 100 for r in f.records) == 34

    land_sea = read(f, "SURFIND.TERREMER")
    assert land_sea.shape == (69, 49)
    assert set(np.unique(land_sea)) <= {0.0, 1.0}

    for record in f.records:
        assert np.isfinite(decode(f.read(record), f.frame)).all(), record.name


@pytest.mark.parametrize(
    "name,field,peak",
    [
        ("regLL_small", "SURFGEOPOTENTIEL", (3991.4, 45.85, 6.875)),  # Mont Blanc
        ("lambert_HN", "SPECSURFGEOPOTEN", (3238.2, 31.51, -6.41)),  # High Atlas
    ],
)
def test_row_order(name, field, peak):
    f = FAFile(sample(name))

    orography = read(f, field) / 9.80665
    j, i = np.unravel_index(orography.argmax(), orography.shape)
    geography = FAGeography(f.frame)
    height, latitude, longitude = peak
    assert orography[j, i] == pytest.approx(height, abs=0.5)
    assert geography.latitudes()[j, i] == pytest.approx(latitude, abs=0.01)
    assert geography.longitudes()[j, i] == pytest.approx(longitude, abs=0.01)


def test_packed_spectral_on_non_square_domain():
    f = FAFile(sample("forecast"))
    assert (f.frame.ndlon, f.frame.ndgl) == (50, 40)
    assert (f.frame.nmsmax, f.frame.nsmax) == (24, 19)
    assert all(f.header(r).ngrib == 2 for r in f.records if f.header(r).spectral)

    # SW corner, centre and NE corner, as transformed by EPyGrAM 2.1.0
    j, i = [0, 14, 28], [0, 19, 38]
    temperature = read(f, "S090TEMPERATURE")
    assert temperature.shape == (29, 39)
    assert temperature[j, i] == pytest.approx([296.898172292, 296.439528823, 299.626498771], abs=1e-8)
    # stored spectrally as the logarithm of surface pressure
    pressure = np.exp(read(f, "SURFPRESSION"))
    assert pressure[j, i] == pytest.approx([100604.239065, 100991.501989, 101169.414147], abs=1e-5)


def test_packed_spectral_with_subtruncation_and_row_order():
    # the low wavenumbers are left unpacked and the others scaled by a power of the Laplacian,
    # which none of the EPyGrAM files does
    f = FAFile(sample("analysis"))
    headers = [f.header(r) for r in f.records]
    assert {(h.sptrunc, h.sppow) for h in headers if h.spectral and h.ngrib} == {(40, 0), (40, 1), (40, 2)}

    orography = read(f, "SPECSURFGEOPOTEN") / 9.80665
    assert orography.shape == (564, 564)
    j, i = np.unravel_index(orography.argmax(), orography.shape)
    geography = FAGeography(f.frame)
    # highest point of the domain is in the Black Forest
    assert orography[j, i] == pytest.approx(1333.3, abs=0.5)
    assert geography.latitudes()[j, i] == pytest.approx(47.9, abs=0.1)
    assert geography.longitudes()[j, i] == pytest.approx(8.0, abs=0.1)

    temperature = read(f, "S087TEMPERATURE")
    assert 260 < temperature.min() < temperature.max() < 300
    # stored spectrally as the logarithm of surface pressure
    pressure = np.exp(read(f, "SURFPRESSION"))
    assert 85000 < pressure.min() < pressure.max() < 106000


@pytest.mark.skipif(not os.environ.get("FA_TEST_EPYGRAM"), reason="set FA_TEST_EPYGRAM=1 to compare with EPyGrAM")
def test_against_epygram(tmp_path):
    epygram = pytest.importorskip("epygram")
    epygram.init_env()
    # EPyGrAM opens files read-write, so work on a copy
    path = shutil.copy(sample("forecast"), tmp_path / "FA")

    resource = epygram.formats.resource(str(path), "r")
    f = FAFile(path)
    for record in f.records:
        field = resource.readfield(record.name)
        if field.spectral:
            field.sp2gp()
        expected = np.asarray(field.getdata(subzone="CI"), dtype=float)
        tolerance = 1e-12 * max(np.abs(expected).max(), 1.0)
        assert np.abs(decode(f.read(record), f.frame) - expected).max() <= tolerance, record.name
    resource.close()
