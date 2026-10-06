"""Decode one raw FA record into a 2-D gridpoint array."""

import math
import struct
from functools import lru_cache

import numpy as np

from .fafile import spectral_mmax

_SURFEX_MISSING = 1e20


def decode(buf, frame):
    """Return the field as float64 with shape (ny, nx), rows south to north, C+I zone only."""
    ngrib, spectral = struct.unpack_from(">2q", buf)
    if spectral:
        if ngrib >= 100:
            raise NotImplementedError("spectral fields packed as GRIB2 are not supported")
        return _trim(_spectral_to_gridpoint(_spectral(buf, ngrib, frame), frame), frame)
    if ngrib >= 100:
        values = _grib2(buf)
    elif ngrib > 0:
        values = _gridpoint_grib0(buf, ngrib, frame)
    else:
        values = _gridpoint_raw(buf, frame)
    return _to_grid(values, frame)


def _to_grid(values, frame):
    if values.size == frame.ndgl * frame.ndlon:
        return _trim(values.reshape(frame.ndgl, frame.ndlon), frame)
    if values.size == frame.ny * frame.nx:
        return values.reshape(frame.ny, frame.nx)
    raise ValueError(f"unexpected number of gridpoint values: {values.size}")


def _trim(grid, frame):
    return np.ascontiguousarray(grid[frame.ndgun - 1 : frame.ndgux, frame.ndlun - 1 : frame.ndlux])


def _gridpoint_raw(buf, frame):
    n_bytes = len(buf) - 16
    for n in (frame.ndgl * frame.ndlon, frame.ny * frame.nx):
        if n_bytes == 8 * n:
            values = np.frombuffer(buf, ">f8", n, 16).astype(np.float64)
            values[values == _SURFEX_MISSING] = np.nan
            return values
        if n_bytes == 8 * ((n + 1) // 2):
            # single precision: two floats per big-endian 64-bit word, in swapped order
            single = np.frombuffer(buf, ">f4", offset=16).reshape(-1, 2)[:, ::-1].reshape(-1)[:n]
            values = single.astype(np.float64)
            values[single == np.float32(_SURFEX_MISSING)] = np.nan
            return values
    raise ValueError(f"unexpected length of an unpacked gridpoint record: {len(buf)} bytes")


def _gridpoint_grib0(buf, ngrib, frame):
    if ngrib >= 2:
        vmin, vmax = struct.unpack_from(">2d", buf, 24)
        values, _ = _grib0(buf, 40, vmin, vmax, frame.ndgl * frame.ndlon)
    else:
        values, _ = _grib0(buf, 24, None, None, frame.ndgl * frame.ndlon)
    return values


def _grib0(buf, start, vmin, vmax, count):
    """Decode the GRIB edition 0 message at ``start``; return (values, message length).

    ``count`` is only used when the message stores zero bits per value.
    """
    if buf[start : start + 4] != b"GRIB":
        raise ValueError("GRIB-0 message not found in FA record")
    sec1 = start + 4
    len1 = int.from_bytes(buf[sec1 : sec1 + 3], "big")
    if buf[sec1 + 3] != 0 or buf[sec1 + 7] & 0xC0:
        raise NotImplementedError("only GRIB edition 0 without grid or bitmap sections is supported")
    sec4 = sec1 + len1
    len4 = int.from_bytes(buf[sec4 : sec4 + 3], "big")
    unused_bits = buf[sec4 + 3] & 15
    nbits = buf[sec4 + 10]
    length = 4 + len1 + len4 + 4
    if buf[start + length - 4 : start + length] != b"7777":
        raise ValueError("corrupted GRIB-0 message in FA record")

    if nbits == 0:
        packed = np.zeros(count, dtype=np.float64)
    else:
        n = ((len4 - 11) * 8 - unused_bits) // nbits
        packed = _unpack_bits(buf, sec4 + 11, nbits, n)

    if vmin is None:
        # standard GRIB scaling: IBM float reference value and binary scale factor
        exponent = int.from_bytes(buf[sec4 + 4 : sec4 + 6], "big")
        if exponent > 32768:
            exponent = 32768 - exponent
        return _ibm_float(buf[sec4 + 6 : sec4 + 10]) + packed * 2.0**exponent, length

    # FA scaling between the stored minimum and maximum; the upper half is computed
    # from the maximum so that both extremes are reproduced exactly
    top = float(2**nbits - 1) if nbits else 1.0
    scale = (vmax - vmin) / top
    values = vmin + scale * packed
    upper = packed >= 2.0 ** (nbits - 1)
    values[upper] = vmax - scale * (top - packed[upper])
    return values, length


def _unpack_bits(buf, offset, nbits, n):
    """Return ``n`` unsigned big-endian ``nbits``-wide integers as float64."""
    if nbits in (8, 16, 32):
        return np.frombuffer(buf, f">u{nbits // 8}", n, offset).astype(np.float64)
    if nbits > 32:
        raise NotImplementedError(f"{nbits} bits per value")

    # the bit layout repeats every `period` bytes, which hold `per_period` values
    common = math.gcd(nbits, 8)
    period, per_period = nbits // common, 8 // common
    groups = -(-n // per_period)
    data = np.frombuffer(buf, np.uint8, offset=offset)
    if data.size < groups * period:
        data = np.concatenate([data, np.zeros(groups * period - data.size, np.uint8)])
    data = data[: groups * period].reshape(groups, period)

    values = np.empty((groups, per_period))
    mask = np.uint64(2**nbits - 1)
    for j in range(per_period):
        first, last = (j * nbits) // 8, ((j + 1) * nbits - 1) // 8
        window = data[:, first].astype(np.uint64)
        for byte in range(first + 1, last + 1):
            window = (window << np.uint64(8)) | data[:, byte]
        values[:, j] = (window >> np.uint64(8 * (last + 1) - (j + 1) * nbits)) & mask
    return values.reshape(-1)[:n]


def _ibm_float(four_bytes):
    sign = -1.0 if four_bytes[0] & 0x80 else 1.0
    exponent = (four_bytes[0] & 0x7F) - 64
    mantissa = int.from_bytes(four_bytes[1:4], "big")
    return sign * mantissa * 2.0**-24 * 16.0**exponent


@lru_cache(maxsize=16)
def _wavenumbers(nsmax, nmsmax):
    """Wavenumbers (n in y, m in x) of the stored coefficients, in file order."""
    mmax = spectral_mmax(nsmax, nmsmax)
    n = np.repeat(np.arange(nsmax + 1), mmax + 1)
    m = np.concatenate([np.arange(k + 1) for k in mmax])
    return n, m


@lru_cache(maxsize=32)
def _packed_split(nsmax, nmsmax, sptrunc, sppow):
    """Positions of the packed and unpacked coefficients, and the factor undoing the packing scale."""
    n, m = _wavenumbers(nsmax, nmsmax)
    packed = m > np.where(n == 0, nmsmax, np.maximum(0, sptrunc - n))
    factor = None
    if sppow > 0:
        # packed coefficients were multiplied by a power of the Laplacian
        factor = (1.0 / (n[packed] ** 2 + m[packed] ** 2).astype(np.float64) ** sppow)[:, None]
    return np.flatnonzero(packed), np.flatnonzero(~packed), factor


@lru_cache(maxsize=16)
def _spectrum_index(nsmax, nmsmax, ny, nx):
    """Flat positions of the coefficients in the (ny, nx // 2 + 1) half spectrum."""
    if nmsmax > nx // 2 or nsmax > ny // 2:
        raise ValueError(f"spectral truncation ({nmsmax}, {nsmax}) too large for a {nx}x{ny} grid")
    n, m = _wavenumbers(nsmax, nmsmax)
    width = nx // 2 + 1
    mirrored = np.flatnonzero((n > 0) & (n <= (ny - 1) // 2))
    return n * width + m, mirrored, (ny - n[mirrored]) * width + m[mirrored]


def _spectral(buf, ngrib, frame):
    """Return the spectral coefficients as an array of shape (number of (n, m) pairs, 4)."""
    n_pairs = _wavenumbers(frame.nsmax, frame.nmsmax)[0].size
    if ngrib <= 0:
        return np.frombuffer(buf, ">f8", 4 * n_pairs, 16).reshape(-1, 4)

    _, sptrunc, sppow = struct.unpack_from(">3q", buf, 16)
    if ngrib >= 2:
        vmin, vmax = struct.unpack_from(">2d", buf, 40)
        start = 56
    else:
        vmin = vmax = None
        start = 40

    # the GRIB message holds the high wavenumbers; the low ones follow it unpacked
    packed, unpacked, factor = _packed_split(frame.nsmax, frame.nmsmax, sptrunc, sppow)
    high, length = _grib0(buf, start, vmin, vmax, 4 * packed.size)
    if high.size != 4 * packed.size:
        raise ValueError("unexpected number of packed spectral coefficients")
    high = high.reshape(-1, 4)
    if factor is not None:
        high *= factor

    low_start = start + length + (-length) % 8
    coefficients = np.empty((n_pairs, 4))
    coefficients[packed] = high
    coefficients[unpacked] = np.frombuffer(buf, ">f8", 4 * unpacked.size, low_start).reshape(-1, 4)
    return coefficients


def _spectral_to_gridpoint(coefficients, frame):
    """Inverse bi-Fourier transform onto the full (ndgl, ndlon) grid."""
    ny, nx = frame.ndgl, frame.ndlon
    index, mirrored, mirrored_index = _spectrum_index(frame.nsmax, frame.nmsmax, ny, nx)
    a0, a1, a2, a3 = coefficients.T

    # the field is real, so only the half of the spectrum with x wavenumber >= 0 is needed
    spectrum = np.zeros(ny * (nx // 2 + 1), dtype=np.complex128)
    spectrum[index] = (a0 - a3) + 1j * (a1 + a2)
    spectrum[mirrored_index] = ((a0 + a3) + 1j * (a2 - a1))[mirrored]
    return np.fft.irfft2(spectrum.reshape(ny, -1), s=(ny, nx)) * (nx * ny)


def _grib2(buf):
    import eccodes

    message = bytes(buf[24:])
    if message[4:8] == b"BIRG":
        # written as 64-bit words on a little-endian machine
        usable = len(message) // 8 * 8
        message = np.frombuffer(message, np.uint8, usable).reshape(-1, 8)[:, ::-1].tobytes()

    handle = eccodes.codes_new_from_message(message)
    try:
        has_bitmap = eccodes.codes_get(handle, "bitmapPresent")
        if has_bitmap:
            eccodes.codes_set(handle, "missingValue", _SURFEX_MISSING)
        values = eccodes.codes_get_values(handle)
        # FA stores some fields in other units than the GRIB2 parameter definition
        factor = 1.0
        if eccodes.codes_is_defined(handle, "FMULTM"):
            factor = eccodes.codes_get(handle, "FMULTM") * 10.0 ** eccodes.codes_get(handle, "FMULTE")
    finally:
        eccodes.codes_release(handle)

    if has_bitmap:
        values[values == _SURFEX_MISSING] = np.nan
    return values / factor
