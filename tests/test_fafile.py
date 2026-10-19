import datetime
import pickle

import pytest
from conftest import sample

from earthkit_data_fa.fafile import FAFile, Record


def test_forecast_frame_and_records():
    f = FAFile(sample("forecast"))
    frame = f.frame

    assert (frame.ndlon, frame.ndgl) == (50, 40)
    assert (frame.nx, frame.ny) == (39, 29)
    assert (frame.nsmax, frame.nmsmax, frame.nlev) == (19, 24, 90)
    assert frame.projection == "lambert"
    assert (frame.dx, frame.dy) == (2500.0, 2500.0)
    assert frame.lon0 == pytest.approx(-51.5)
    assert frame.lat0 == pytest.approx(5.0)
    assert frame.pref == 101325.0
    assert len(frame.a) == len(frame.b) == 91
    assert frame.a[0] == 0.0 and frame.b[-1] == 1.0

    assert len(f.records) == 134
    assert sum(f.header(r).spectral for r in f.records) == 91
    assert len({r.name for r in f.records}) == 134
    assert not any(r.name.startswith(("CADRE", "DATE", "DATX")) for r in f.records)

    assert f.base_datetime == datetime.datetime(2017, 2, 14)
    assert f.step == datetime.timedelta(hours=2)
    assert f.validity == {"time_range_indicator": 10, "cumulation_start": 0, "timestep": 0}


def test_analysis_validity():
    # one field of the same run, at the initial time
    f = FAFile(sample("VGeometry_hybridP"))
    assert f.base_datetime == datetime.datetime(2017, 2, 14)
    assert f.step == datetime.timedelta(0)
    assert f.validity["time_range_indicator"] == 0


def test_validity_without_extended_date():
    # written without the DATX-DES-DONNEES article, date is a climatology placeholder
    f = FAFile(sample("clim"))
    assert f.base_datetime == datetime.datetime(1, 9, 15)
    assert f.step == datetime.timedelta(0)
    assert len(f.records) == 40


def test_opening_reads_no_record(monkeypatch):
    # a read per record is what made a large file slow to open on a network filesystem
    reads = []
    original = FAFile._pread

    def counting(self, length, offset):
        reads.append(offset)
        return original(self, length, offset)

    monkeypatch.setattr(FAFile, "_pread", counting)

    f = FAFile(sample("forecast"))
    assert len(f.records) == 134
    assert not set(reads) & {r.offset for r in f.records}

    record = next(r for r in f.records if r.name == "S090TEMPERATURE")
    del reads[:]
    assert (f.header(record).spectral, f.header(record).ngrib, f.header(record).nbits) == (True, 2, 16)
    assert reads == [record.offset]


def test_short_article_that_is_not_a_field_is_left_out():
    f = FAFile(sample("forecast"))
    articles = f._read_index()
    articles["NOT A FIELD"] = articles["DATE-DES-DONNEES"]
    assert "NOT A FIELD" not in {r.name for r in f._read_records(articles)}


def test_header_of_an_article_that_is_not_a_field():
    f = FAFile(sample("forecast"))
    with pytest.raises(ValueError, match="does not hold a field"):
        f.header(Record("DATE-DES-DONNEES", *f._read_index()["DATE-DES-DONNEES"]))


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
