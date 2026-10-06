"""Sample FA files are looked up relative to the project root; tests skip when a file is absent."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "analysis": ROOT / "data/input/2019/01/01/analysis",
    "first_guess": ROOT / "data/input/2019/01/01/first_guess",
    "clim": ROOT / "libs/pyfalib/test/Const.Clim.09",
    "forecast": ROOT / "libs/EPyGrAM/tests/data/formats/FA",
    "lfi": ROOT / "libs/EPyGrAM/tests/data/formats/LFI",
}
GEOMETRIES = ROOT / "libs/EPyGrAM/tests/data/geometries"


def sample(name):
    path = FILES[name] if name in FILES else GEOMETRIES / f"{name}.fa"
    if not path.exists():
        pytest.skip(f"sample file not available: {path}")
    return str(path)
