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
    assert sum(r.ngrib >= 100 for r in f.records) == 34

    land_sea = read(f, "SURFIND.TERREMER")
    assert land_sea.shape == (69, 49)
    assert set(np.unique(land_sea)) <= {0.0, 1.0}

    for record in f.records:
        assert np.isfinite(decode(f.read(record), f.frame)).all(), record.name


def test_packed_spectral_and_row_order():
    f = FAFile(sample("analysis"))

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
