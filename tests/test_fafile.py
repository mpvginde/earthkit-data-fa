import datetime
import pickle

import pytest
from conftest import sample

from earthkit_data_fa.fafile import FAFile


def test_analysis_frame_and_records():
    f = FAFile(sample("analysis"))
    frame = f.frame

    assert (frame.ndlon, frame.ndgl) == (576, 576)
    assert (frame.nx, frame.ny) == (564, 564)
    assert (frame.nsmax, frame.nmsmax, frame.nlev) == (287, 287, 87)
    assert frame.projection == "lambert"
    assert (frame.dx, frame.dy) == (1300.0, 1300.0)
    assert frame.lon0 == pytest.approx(4.55)
    assert frame.lat0 == pytest.approx(50.8)
    assert frame.pref == 101325.0
    assert len(frame.a) == len(frame.b) == 88
    assert frame.a[0] == 0.0 and frame.b[-1] == 1.0

    assert len(f.records) == 1251
    assert sum(r.spectral for r in f.records) == 524
    assert len({r.name for r in f.records}) == 1251
    assert not any(r.name.startswith(("CADRE", "DATE", "DATX")) for r in f.records)

    assert f.base_datetime == datetime.datetime(2019, 1, 1, 3)
    assert f.step == datetime.timedelta(0)


def test_forecast_validity():
    f = FAFile(sample("first_guess"))
    assert f.base_datetime == datetime.datetime(2019, 1, 1, 0)
    assert f.step == datetime.timedelta(hours=3)
    assert f.validity["cumulation_start"] == 7200


def test_validity_without_extended_date():
    # written without the DATX-DES-DONNEES article, date is a climatology placeholder
    f = FAFile(sample("clim"))
    assert f.base_datetime == datetime.datetime(1, 9, 15)
    assert f.step == datetime.timedelta(0)
    assert len(f.records) == 40


def test_pickle_reopens_file():
    f = FAFile(sample("forecast"))
    record = f.records[0]
    expected = f.read(record)
    clone = pickle.loads(pickle.dumps(f))
    assert clone.frame == f.frame
    assert clone.read(record) == expected


def test_plain_lfi_is_rejected():
    with pytest.raises(ValueError, match="without an FA frame"):
        FAFile(sample("lfi"))


def test_global_geometry_is_rejected():
    with pytest.raises(NotImplementedError, match="global"):
        FAFile(sample("gaussC1"))


def test_other_file_is_rejected():
    with pytest.raises(ValueError, match="not an FA/LFI file"):
        FAFile(__file__)
