"""Sample files come from the EPyGrAM repository and are downloaded once into ``.cache``.

What none of those files contains is tested on local files, looked up relative to the project root.
Tests skip when a file is not available.
"""

import os
import urllib.request
from pathlib import Path

import pytest

import earthkit_data_fa.decode

EPYGRAM_TAG = "2.1.0"
EPYGRAM_DATA = f"https://raw.githubusercontent.com/UMR-CNRM/EPyGrAM/{EPYGRAM_TAG}/tests/data/"
FORMATS = {"forecast": "formats/FA", "lfi": "formats/LFI"}
CACHE = Path(__file__).resolve().parents[1] / ".cache" / f"epygram-{EPYGRAM_TAG}"

ROOT = Path(__file__).resolve().parents[2]
LOCAL = {
    # spectral fields packed with a sub-truncation and a power of the Laplacian, names cut by the level prefix
    "analysis": ROOT / "data/input/2019/01/01/analysis",
    # records packed as GRIB2, no DATX-DES-DONNEES article
    "clim": ROOT / "libs/pyfalib/test/Const.Clim.09",
}

_download_error = None


def sample(name):
    if name in LOCAL:
        path = LOCAL[name]
        if not path.exists():
            pytest.skip(f"sample file not available: {path}")
        return str(path)

    relative = FORMATS.get(name, f"geometries/{name}.fa")
    path = CACHE / relative
    if not path.exists():
        _download(EPYGRAM_DATA + relative, path)
    return str(path)


def _download(url, path):
    global _download_error
    # after a first failure the other downloads are not attempted
    if _download_error is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(f"{path}.{os.getpid()}")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(path)
            return
        except OSError as error:
            partial.unlink(missing_ok=True)
            _download_error = error
    pytest.skip(f"cannot download {url}: {_download_error}")


@pytest.fixture
def decode_calls(monkeypatch):
    calls = []
    original = earthkit_data_fa.decode.decode

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(earthkit_data_fa.decode, "decode", counting)
    return calls
