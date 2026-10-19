# earthkit-data-fa

An [earthkit-data](https://github.com/ecmwf/earthkit-data) source plugin for reading **FA
files** — the self-describing binary format written by the ALADIN/ARPEGE/AROME/ALARO
limited-area NWP models — directly as earthkit `Field`s, with correct CRS and grid metadata,
and without decoding any values until they are actually requested. It also gives
anemoi-datasets an `fa` source, see [Building anemoi datasets](#building-anemoi-datasets).

## earthkit-data and its plugin system

earthkit-data is ECMWF's Python library for reading, selecting and converting
meteorological data (GRIB, NetCDF, CSV, ...) through one interface:

```python
import earthkit.data as ekd

ds = ekd.from_source("file", "data.grib")
fl = ds.to_fieldlist()
```

`from_source(name, ...)` dispatches on `name` to a **source**: a class that knows how to turn
some input (a path, a URL, a request) into data. Most sources are built in, but third parties
can add their own as a pip-installable plugin, with no change to earthkit-data itself. A
plugin package registers its source class under the `earthkit.data.sources` entry point:

```toml
[project.entry-points."earthkit.data.sources"]
fa = "earthkit_data_fa:FASource"
```

Once installed, `ekd.from_source("fa", path)` resolves to `FASource` exactly as if it were
built in. This plugin is deliberately **only a source**, not a format reader registered with
`"file"`: earthkit-data's `"file"` source auto-detects built-in formats by magic bytes, but has
no equivalent plugin hook for new readers, so FA files are opened by name rather than by
auto-detection.

`FASource.mutate()` parses the file's index and frame (about ten reads near the start of the
file, whatever the number of fields) and returns a `SimpleFieldList` of lazy `Field` objects.
Each field's data handler only reads and decodes its own record when `.to_numpy()`, `.values`
or similar is called — selecting, listing or building an xarray dataset touches none of the
810 MB a typical file holds. The packing flags of a field (`metadata.spectral`,
`metadata.ngrib`, `metadata.nbits`) sit at the start of its record and are read when first
asked for, so selecting on one of them costs one small read per field.

## Installation

```bash
pip install -e /path/to/earthkit-data-fa
# or, with uv:
uv pip install -e /path/to/earthkit-data-fa
```

There is no PyPI release; install the checkout directly. This registers the `fa` source for
any environment that also has `earthkit-data` installed — no further setup, and nothing to
import explicitly (`ekd.from_source("fa", ...)` finds it through the entry point).

## Basic example: to a FieldList, and to xarray

```python
import earthkit.data as ekd

fl = ekd.from_source("fa", "data/input/2019/01/01/analysis").to_fieldlist()
len(fl)
# 1251
fl[0]
# Field(PROFTEMPERATURE, 2019-01-01 03:00:00, 2019-01-01 03:00:00, 0:00:00, None, unknown, None, lambert)

temperature = fl.sel({"parameter.variable": "TEMPERATURE", "vertical.level_type": "hybrid"})
ds = temperature.to_xarray()
```

```
<xarray.Dataset> Size: 226MB
Dimensions:      (level: 87, y: 564, x: 564)
Coordinates:
  * level        (level) int64 696B 1 2 3 4 5 6 7 8 ... 80 81 82 83 84 85 86 87
    latitude     (y, x) float64 3MB ...
    longitude    (y, x) float64 3MB ...
Dimensions without coordinates: y, x
Data variables:
    TEMPERATURE  (level, y, x) float64 221MB ...
Attributes:
    Conventions:  CF-1.8
    institution:  ECMWF
```

`sel()` and `to_xarray()` are earthkit-data's own, not reimplemented by this plugin; nothing
above decoded a value until `to_xarray()` indexed into one of the 87 levels.

## Getting at the FA frame metadata

Every field also carries the raw information from the file's FA "frame" — the header record
that holds the grid and projection parameters — reachable through `field.metadata(key)` (or
equivalently `field.get("metadata." + key)`):

```python
f = fl[0]

f.metadata("name")         # 'PROFTEMPERATURE'  (the original, un-split FA field name)
f.metadata("nsmax")        # 287                (spectral truncation)
f.metadata("dx")           # 1300.0             (grid spacing, metres)
f.metadata("projection")   # 'lambert'

f.geography.projection().to_proj_string()
# '+proj=lcc +lat_1=50.8 +lat_2=50.8 +lat_0=50.8 +lon_0=4.55 +units=m +a=6371229.0 +b=6371229.0 +no_defs'
```

`metadata()` first checks the field's own record (`name`, `spectral`, `ngrib`, `nbits`), then
falls through to whatever attribute of that name exists on the file's `Frame` — so every value
read from the FA header (`ndlun`, `ndlux`, `lon0`, `lat0`, the hybrid-level coefficients `a`
and `b`, and so on) is available this way, with no separate list of "supported" keys to keep
up to date. Unlike GRIB keys, this is FA-specific raw metadata: there is no shortName,
paramId or units behind it.

Raw metadata is never pulled into `to_xarray()` output on its own — pass the keys you want
explicitly:

```python
fl.to_xarray(variable_attrs=["metadata.dx"], global_attrs=["metadata.projection"])
```

## Building anemoi datasets

The package also registers an `fa` source with
[anemoi-datasets](https://github.com/ecmwf/anemoi-datasets), so that a recipe reads FA files
directly:

```yaml
dates:
  start: 2019-01-01T00:00:00
  end: 2019-01-02T21:00:00
  frequency: 3h

input:
  join:
    - fa:
        path: /data/AO13/{date:strftime(%Y/%m)}/BEHRA_AO13_{date:strftime(%Y%m%d_%H)}
        parameter.variable: [TEMPERATURE, VENT_ZONAL, VENT_MERIDIEN, HUMI.SPECIFIQUE, GEOPOTENTIEL]
        vertical.level_type: pressure
        vertical.level: [500, 850]
    - fa:
        path: /data/AO13/{date:strftime(%Y/%m)}/BEHRA_AO13_{date:strftime(%Y%m%d_%H)}
        parameter.variable: [SURFPRESSION, MSLPRESSURE]
```

```bash
anemoi-datasets create recipe.yaml dataset.zarr
```

- **`path`** is one path or a list of them. It may hold glob wildcards and
  `{date:strftime(...)}`, which is replaced by each of the dates being built, so only the
  files of those dates are opened. A file that does not exist raises `FileNotFoundError`.
- **The other keys select fields**, as `FieldList.sel()` does. Use the earthkit-data names
  shown above, or `metadata.name` for the raw FA record name (`S087TEMPERATURE`). The short
  names of GRIB recipes (`param`, `levtype`, `levelist`) work too, with a deprecation warning
  from anemoi-transform. A bare `name` cannot be used: the anemoi registry takes it.
- **Only the selected fields are decoded**, when the dataset is written.
- **Variables are named `<variable>_<level>`**: `TEMPERATURE_850`, `SURFPRESSION`. A file
  often holds one variable on several types of level, and the name does not tell them apart:
  without `vertical.level_type`, the 500 hPa and the 500 m temperature are both
  `TEMPERATURE_500`.

The dataset keeps the FA names, and its variables have the units `dimensionless`, which is
what earthkit-data reports for a field without units. Both are set in the recipe, with the
`set_metadata` and `rename` filters of anemoi-transform:

```yaml
input:
  join:
    - pipe:
        - fa:
            path: ...
            parameter.variable: TEMPERATURE
            vertical.level_type: pressure
            vertical.level: [500, 850]
        - set_metadata:          # t_500 and t_850, in kelvin
            metadata:
              parameter.variable: t
              parameter.units: K
    - pipe:
        - fa:
            path: ...
            parameter.variable: SURFPRESSION
        - set_metadata:
            metadata:
              parameter.units: Pa
        - rename:                # sp
            param:
              SURFPRESSION: sp
```

### Accumulations

The `accumulate` source of anemoi-datasets works on top of `fa`, for files whose accumulated
records each cover a fixed period ending at their valid time. This gives the precipitation of
the last 6 hours from files that hold 3 hours each:

```yaml
- accumulate:
    period: 6h
    covering:
      auto: [["*", "0-3/3-6/6-9/9-12/12-15/15-18/18-21/21-24"]]
    source:
      fa:
        path: /data/AO13/{date:strftime(%Y/%m)}/BEHRA_AO13_{date:strftime(%Y%m%d_%H)}
        parameter.variable: SURFPREC.EAU.GEC
```

- **`covering` says what the files hold.** `"*"` stands for "no base time": the files are found
  by valid time. The steps are hours counted from 00 UTC of each day, so the line above
  describes 3 h accumulations ending at 03, 06, ... UTC. The short form `auto: 3h` does not
  do that on this branch of anemoi-datasets, it always describes 1 h steps.
- **It is taken on trust.** An FA record does not say over which period it is accumulated, so
  nothing checks the `covering` against the files. Only a file that is missing is noticed.
- **A file is read for the end of each interval**, also before the first date: the 6 h ending
  at 03 UTC need the file valid at 00 UTC, and `FileNotFoundError` is raised if it is not there.
- The result is recorded in the dataset as an accumulation over `period`. It keeps the FA
  name, so `set_metadata`, `rename` and the other filters go after it in a `pipe`, as above.

Archives accumulated since the start of a forecast, where a base time is needed to find a
record, are not supported.

### Requirements

anemoi-datasets and anemoi-transform built on earthkit-data 1.x. In October 2026 that is the
branch `feat/earthkit-data-1.0` of both, the draft pull requests
[anemoi-datasets #695](https://github.com/ecmwf/anemoi-datasets/pull/695) and
[anemoi-transform #317](https://github.com/ecmwf/anemoi-transform/pull/317). The releases up to
anemoi-datasets 0.5.45 and anemoi-transform 0.5.0 require `earthkit-data<1` and cannot be
installed next to this plugin.

```bash
git clone -b feat/earthkit-data-1.0 https://github.com/ecmwf/anemoi-transform
git clone -b feat/earthkit-data-1.0 https://github.com/ecmwf/anemoi-datasets
uv pip install -e anemoi-transform -e "anemoi-datasets[create]" -e earthkit-data-fa
```

This was tested with earthkit-data 1.2.3, at commits `635e26a` of anemoi-transform and
`7666c35` of anemoi-datasets.

On the ECMWF HPC, the `ecmwf-toolbox` module sets `FINDLIBS_DISABLE_PACKAGE`, which makes
Python load the system eccodes, eckit and their dependencies instead of the ones in the
wheels. Among them is a PROJ library that clashes with the one of `pyproj`: a build does its
work, then the process crashes when it exits. Unset the variable for anything that uses
anemoi-datasets:

```bash
env -u FINDLIBS_DISABLE_PACKAGE anemoi-datasets create recipe.yaml dataset.zarr
```

### Limits

- Wind components are returned as they are stored, along the grid axes or along east and
  north depending on how the file was produced. Nothing is rotated. anemoi-transform has a
  `rotate_winds` filter, which has not been tried on FA fields.
- FA records carry no accumulation period. A field read with `fa` alone is not recorded as
  an accumulation, whatever it holds; under `accumulate` the period comes from the recipe (see
  "Accumulations").
- A field that went through `set_metadata` no longer answers raw FA metadata
  (`metadata.name`, `metadata.gridType`, ...): earthkit-data hides it once a component of the
  field is changed. One effect is that the `resolution` of the dataset is then a meaningless
  number instead of `lambert`.
- The fields come out as the file holds them. In the `+0000` files this was tested on, the
  screen-level diagnostics (`CLSTEMPERATURE`, `CLSVENT.ZONAL`, ...) are zero everywhere, and
  so they are in the dataset.

## Design choices

This plugin favours simplicity, correct metadata and laziness over completeness:

- **Own decoder, no FA library dependency.** Values are decoded by a small pure-numpy
  implementation (`decode.py`), written against the reference C code in
  [Rfa](https://github.com/harphub/Rfa) and its format documentation, not by linking
  [pyfalib](https://github.com/harphub/pyfalib) or [EPyGrAM](https://github.com/UMR-CNRM/EPyGrAM).
  `pyfalib` doesn't build as checked out and its spectral decoding has known bugs; EPyGrAM is
  correct but depends on compiled Fortran libraries and opens files read-write. eccodes
  (already a dependency of earthkit-data) is only used for the rare field stored as an
  embedded GRIB2 message.
- **Spectral fields are transformed to gridpoint eagerly on read, not exposed raw.** An FA file
  mixes gridpoint and spectral fields (upper-air fields are typically spectral); decoding both
  to gridpoint means every field in a file shares one grid and CRS, and `to_xarray()` /
  `sel()` / plotting all work uniformly. The field's packing is still visible as
  `metadata("spectral")`.
- **Explicit `"fa"` source name, not auto-detection via `"file"`.** earthkit-data has no reader
  plugin mechanism to hook into (only sources), so this is also the only workable route.
- **Variable names are parsed from the FA field name, with level-prefix truncation undone,
  and no FA→GRIB parameter table.** `S087TEMPERATURE` becomes variable `TEMPERATURE`, level 87,
  level type `hybrid`; `P85000VENT_MERID` becomes `VENT_MERIDIEN`, level 850, type `pressure`
  — both the generic 16-character cut and the model's own 12-character abbreviations are
  restored against a short list of known upper-air names (`field.py:_MODEL_NAMES`), not a full
  parameter table. There is deliberately no mapping to GRIB `shortName`/`paramId`/units.
- **Spectral `SURFPRESSION` is returned as surface pressure in Pa**, not as the logarithm the
  model stores it as, so it reads like every other pressure field.
- **Limited-area files only.** ARPEGE/ALADIN global (Gaussian) geometry, rotated/tilted
  Mercator and academic (no-projection) geometries raise `NotImplementedError` rather than
  being guessed at.
