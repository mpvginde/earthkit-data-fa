"""Index, frame and raw record access for FA files. Nothing here decodes field values."""

import datetime
import math
import os
import threading
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS = 6371229.0

_HOLE = b" " * 16
_END_OF_INDEX = b"**FIN D'INDEX **"
_FRAME_ARTICLES = (
    "CADRE-DIMENSIONS",
    "CADRE-FRANKSCHMI",
    "CADRE-REDPOINPOL",
    "CADRE-SINLATITUD",
    "CADRE-FOCOHYBRID",
    "DATE-DES-DONNEES",
    "DATX-DES-DONNEES",
)
# GRIB1 code table 4, as used in DATE-DES-DONNEES[5]
_TIME_UNIT_SECONDS = {0: 60, 1: 3600, 2: 86400, 254: 1}
_EPS = 1e-10


@dataclass(frozen=True)
class Frame:
    """Horizontal and vertical geometry shared by all the fields of a file."""

    nsmax: int  # spectral truncation in y
    nmsmax: int  # spectral truncation in x
    ndgl: int  # stored rows, extension zone included
    ndlon: int  # stored columns, extension zone included
    nlev: int
    ndlun: int  # C+I zone bounds, 1-based and inclusive
    ndlux: int
    ndgun: int
    ndgux: int
    projection: str  # lambert, mercator, polar_stereographic or regular_ll
    lon0: float  # projection reference point, degrees
    lat0: float
    anchor_lon: float  # geographical position (degrees) of the grid point at
    anchor_lat: float  # fractional C+I index (anchor_i, anchor_j)
    anchor_i: float
    anchor_j: float
    dx: float  # metres, or degrees for regular_ll
    dy: float
    pref: float  # reference pressure, Pa
    a: tuple  # hybrid half-level coefficients in Pa, model top first
    b: tuple

    @property
    def nx(self):
        return self.ndlux - self.ndlun + 1

    @property
    def ny(self):
        return self.ndgux - self.ndgun + 1


@dataclass(frozen=True)
class Record:
    """Location of one field in the file."""

    name: str
    offset: int  # bytes
    length: int  # bytes


@dataclass(frozen=True)
class Header:
    """Packing description at the start of a record."""

    ngrib: int
    spectral: bool
    nbits: int | None
    sptrunc: int | None
    sppow: int | None


class FAFile:
    def __init__(self, path):
        self.path = os.fspath(path)
        self._fd = None
        self._lock = threading.Lock()
        self._headers = {}

        articles = self._read_index()
        self.frame = self._read_frame(articles)
        self.base_datetime, self.step, self.validity = self._read_validity(articles)
        self.records = self._read_records(articles)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fd"] = None
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __del__(self):
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)

    def _pread(self, length, offset):
        if self._fd is None:
            with self._lock:
                if self._fd is None:
                    self._fd = os.open(self.path, os.O_RDONLY)
        return os.pread(self._fd, length, offset)

    def read(self, record):
        """Return the raw bytes of one record."""
        return self._pread(record.length, record.offset)

    def header(self, record):
        """Return the packing description of a record, read from the file on first use."""
        if record.name not in self._headers:
            header = self._read_header(record)
            if header is None:
                raise ValueError(f"{self.path}: article {record.name!r} does not hold a field")
            self._headers[record.name] = header
        return self._headers[record.name]

    def _read_index(self):
        """Return {article name: (byte offset, byte length)}, holes excluded."""
        header = np.frombuffer(self._pread(22 * 8, 0), ">i8")
        if header.size != 22 or header[1] != 16 or header[3] != 22:
            raise ValueError(f"{self.path}: not an FA/LFI file")

        sector_bytes = int(header[0]) * 8
        n_articles = int(header[5])
        per_sector = int(header[12]) or sector_bytes // 16
        address_shift = max(int(header[19]), 1) * sector_bytes

        articles = {}
        name_offset = sector_bytes
        for k in range(math.ceil(n_articles / per_sector)):
            if k > 0:
                # sector numbers of the extra name sectors sit at the end of sector 1
                (sector,) = np.frombuffer(self._pread(8, sector_bytes - 8 * k), ">i8")
                name_offset = (int(sector) - 1) * sector_bytes
            n = min(per_sector, n_articles - k * per_sector)
            names = self._pread(16 * n, name_offset)
            address = np.frombuffer(self._pread(16 * n, name_offset + address_shift), ">i8").reshape(n, 2)
            for i in range(n):
                name = names[16 * i : 16 * i + 16]
                if name in (_HOLE, _END_OF_INDEX):
                    continue
                length, position = address[i]
                articles[name.decode("latin-1").rstrip()] = (8 * (int(position) - 1), 8 * int(length))
        return articles

    def _article(self, articles, name, dtype):
        offset, length = articles[name]
        return np.frombuffer(self._pread(length, offset), dtype)

    def _read_frame(self, articles):
        if "CADRE-DIMENSIONS" not in articles:
            raise ValueError(f"{self.path}: LFI file without an FA frame")

        nsmax, ndgl, ndlon, nlev, ktyptr = (int(v) for v in self._article(articles, "CADRE-DIMENSIONS", ">i8")[:5])
        if ktyptr > 0:
            raise NotImplementedError(f"{self.path}: global (ARPEGE) geometry is not supported")

        if self._article(articles, "CADRE-FRANKSCHMI", ">f8")[3] == -1:
            raise NotImplementedError(f"{self.path}: academic geometry (no map projection) is not supported")

        ndlun, ndlux, ndgun, ndgux = (int(v) for v in self._article(articles, "CADRE-REDPOINPOL", ">i8")[2:6])
        nx = ndlux - ndlun + 1
        ny = ndgux - ndgun + 1

        s = self._article(articles, "CADRE-SINLATITUD", ">f8")
        if int(s[0]) == -2:
            raise NotImplementedError(f"{self.path}: rotated/tilted Mercator is not supported")

        if int(s[0]) != 0:
            # centre of the C+I zone is given
            rpk, lon0, lat0, dx, dy = s[1], s[2], s[3], s[6], s[7]
            anchor = (s[4], s[5], (nx - 1) / 2, (ny - 1) / 2)
        else:
            # old layout: only the SW corner of the C+I zone is given
            rpk, lon0, lat0, dx, dy = s[9], s[7], s[8], s[14], s[15]
            anchor = (s[3], s[4], 0.0, 0.0)

        if int(rpk) == -9:
            projection = "regular_ll"
            dx, dy = math.degrees(dx), math.degrees(dy)
        elif abs(rpk) < _EPS:
            projection = "mercator"
        elif abs(rpk) > 1 - _EPS:
            projection = "polar_stereographic"
        else:
            projection = "lambert"

        levels = self._article(articles, "CADRE-FOCOHYBRID", ">f8")
        pref = float(levels[0])

        return Frame(
            nsmax=nsmax,
            nmsmax=-ktyptr,
            ndgl=ndgl,
            ndlon=ndlon,
            nlev=nlev,
            ndlun=ndlun,
            ndlux=ndlux,
            ndgun=ndgun,
            ndgux=ndgux,
            projection=projection,
            lon0=math.degrees(lon0),
            lat0=math.degrees(lat0),
            anchor_lon=math.degrees(anchor[0]),
            anchor_lat=math.degrees(anchor[1]),
            anchor_i=anchor[2],
            anchor_j=anchor[3],
            dx=float(dx),
            dy=float(dy),
            pref=pref,
            a=tuple(float(v) * pref for v in levels[1 : nlev + 2]),
            b=tuple(float(v) for v in levels[nlev + 2 : 2 * nlev + 3]),
        )

    def _read_validity(self, articles):
        date = [int(v) for v in self._article(articles, "DATE-DES-DONNEES", ">i8")]
        extended = [0] * 11
        if "DATX-DES-DONNEES" in articles:
            extended = [int(v) for v in self._article(articles, "DATX-DES-DONNEES", ">i8")]

        unit = _TIME_UNIT_SECONDS.get(date[5])
        if extended[0] == 1:
            seconds_of_day, step, cumulation_start, timestep = extended[2], extended[3], extended[4], extended[6]
        else:
            seconds_of_day = date[3] * 3600 + date[4] * 60
            step = date[6] * unit if unit else None
            cumulation_start = date[9] * unit if unit else None
            timestep = None

        try:
            base = datetime.datetime(date[0], date[1], date[2]) + datetime.timedelta(seconds=seconds_of_day)
        except ValueError:
            # climatology files carry placeholder dates
            base = None

        validity = {
            "time_range_indicator": date[8],
            "cumulation_start": cumulation_start,
            "timestep": timestep,
        }
        return base, None if step is None else datetime.timedelta(seconds=step), validity

    def _read_records(self, articles):
        # Telling whether an article holds a field takes a read at its position. One such read
        # per record, scattered over the file, is what opening a large file would cost, so only
        # an article too short to hold one bit per grid point is checked here. A longer one is
        # taken to be a field, and header() reads it when first asked.
        unchecked = self.frame.nx * self.frame.ny // 8

        records = []
        for name, (offset, length) in articles.items():
            if name in _FRAME_ARTICLES or length < 16:
                continue
            record = Record(name, offset, length)
            if length < unchecked:
                header = self._read_header(record)
                if header is None:
                    continue
                self._headers[name] = header
            records.append(record)
        return records

    def _read_header(self, record):
        """Return the Header of a record, or None when the article does not hold a field."""
        frame = self.frame
        head = self._pread(min(record.length, 64), record.offset)
        words = np.frombuffer(head[: len(head) // 8 * 8], ">i8")
        if len(words) < 3:
            return None
        ngrib, spectral = int(words[0]), int(words[1])
        if spectral not in (0, 1):
            return None

        nbits = sptrunc = sppow = None
        if ngrib <= 0:
            n_bytes = record.length - 16
            if spectral:
                is_field = n_bytes == 32 * spectral_count(frame.nsmax, frame.nmsmax)
            else:
                gridpoint_sizes = {frame.ndlon * frame.ndgl, frame.nx * frame.ny}
                is_field = any(n_bytes in (8 * n, 8 * math.ceil(n / 2)) for n in gridpoint_sizes)
        else:
            nbits = int(words[2])
            if spectral and len(words) >= 5:
                sptrunc, sppow = int(words[3]), int(words[4])
            if ngrib >= 100:
                is_field = b"GRIB" in head[24:32] or b"BIRG" in head[24:32]
            else:
                start = 24 + (16 if spectral else 0) + (16 if ngrib >= 2 else 0)
                is_field = head[start : start + 4] == b"GRIB"
        return Header(ngrib, bool(spectral), nbits, sptrunc, sppow) if is_field else None


def spectral_mmax(nsmax, nmsmax):
    """Largest x wavenumber kept for each y wavenumber n = 0..nsmax (elliptic truncation)."""
    n = np.arange(nsmax + 1)
    return np.floor(nmsmax * np.sqrt(np.abs(1.0 - (n / nsmax) ** 2)) + 1e-10).astype(np.int64)


def spectral_count(nsmax, nmsmax):
    """Number of (n, m) wavenumber pairs stored for a spectral field."""
    if nsmax <= 0:
        return 0
    return int((spectral_mmax(nsmax, nmsmax) + 1).sum())
