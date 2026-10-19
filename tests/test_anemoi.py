"""The anemoi-datasets source, on the two samples of one run and grid: steps 0 and 2 h."""

import datetime
import os
import types

import earthkit.data as ekd
import numpy as np
import pytest
import yaml
from conftest import sample

if "FINDLIBS_DISABLE_PACKAGE" in os.environ:
    # anemoi-datasets then loads the system libraries, among them a PROJ that clashes with the
    # one of pyproj: the tests pass, but the process crashes when it exits
    pytest.skip("FINDLIBS_DISABLE_PACKAGE is set, unset it to run the anemoi tests", allow_module_level=True)

datasets = pytest.importorskip("anemoi.datasets")

START, END = datetime.datetime(2017, 2, 14, 0), datetime.datetime(2017, 2, 14, 2)


@pytest.fixture
def dated(tmp_path):
    """Path pattern of the two samples, linked under names that hold their valid time."""
    for name, date in (("VGeometry_hybridP", START), ("forecast", END)):
        (tmp_path / f"fa_{date:%Y%m%d%H}").symlink_to(sample(name))
    return str(tmp_path / "fa_{date:strftime(%Y%m%d%H)}")


def recipe(start, end, *sources):
    return {
        "dates": {"start": start.isoformat(), "end": end.isoformat(), "frequency": "2h"},
        "input": {"join": [{"fa": source} for source in sources]},
    }


def accumulate(start, end, period, source):
    """Recipe that accumulates over ``period`` the files, each taken to hold the 2 h before its valid time."""
    steps = "/".join(f"{hour}-{hour + 2}" for hour in range(0, 24, 2))
    block = {"period": period, "covering": {"auto": [["*", steps]]}, "source": {"fa": source}}
    return recipe(start, end) | {"input": {"accumulate": block}}


def temperature(name):
    """S090TEMPERATURE of a sample, read with the plugin alone."""
    return ekd.from_source("fa", sample(name)).to_fieldlist().sel({"metadata.name": "S090TEMPERATURE"})[0]


def process(field):
    """What a dataset would record about the time processing of a field."""
    from anemoi.transform.variables import Variable

    variable = Variable.from_field(field.get("labels.name"), field).as_dict()
    return variable.get("process"), variable.get("period")


def select(recipe):
    """The named fields of the first group of dates, as the dataset builder receives them."""
    from anemoi.datasets.create.input.builder import InputBuilder
    from anemoi.datasets.create.input.context import Context
    from anemoi.datasets.create.recipe import Recipe
    from anemoi.datasets.dates.groups import Groups

    class NamingContext(Context):
        def create_result(self, argument, data):
            return self.naming(data)

    recipe = Recipe(**recipe)
    group = next(iter(Groups(recipe.dates, recipe.build.group_by)))
    return InputBuilder(recipe.input, recipe.data_sources).select(NamingContext(recipe), group)


def test_registered():
    from anemoi.datasets.create.sources import source_registry

    from earthkit_data_fa.anemoi_source import FASource

    assert source_registry.lookup("fa") is FASource


def test_selection(decode_calls):
    # S090TEMPERATURE is in both files: only the one valid at the requested date is kept
    path = [sample("VGeometry_hybridP"), sample("forecast")]
    fields = select(
        recipe(
            END,
            END,
            {"path": path, "param": "TEMPERATURE", "levtype": "hybrid", "levelist": [1, 90]},
            {"path": path, "parameter.variable": ["SURFPRESSION", "SURFTEMPERATURE"]},
            {"path": path, "metadata.name": "S045TEMPERATURE"},
        )
    )

    assert sorted(f.get("labels.name") for f in fields) == [
        "SURFPRESSION",
        "SURFTEMPERATURE",
        "TEMPERATURE_1",
        "TEMPERATURE_45",
        "TEMPERATURE_90",
    ]
    assert {f.get("time.valid_datetime") for f in fields} == {END}
    assert decode_calls == []


def test_one_file_per_date(dated):
    fields = select(recipe(START, END, {"path": dated, "param": "TEMPERATURE", "levelist": 90}))

    assert sorted(f.get("time.valid_datetime") for f in fields) == [START, END]


def test_missing_file(dated):
    with pytest.raises(FileNotFoundError, match="fa_2017021404"):
        select(recipe(START, END + datetime.timedelta(hours=2), {"path": dated}))


def test_dataset(dated, tmp_path):
    from anemoi.datasets.create.tasks import run_task

    path = str(tmp_path / "dataset.zarr")
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.dump(recipe(START, END, {"path": dated, "param": "TEMPERATURE", "levelist": 90})))
    run_task("init", recipe=str(recipe_path), path=path, overwrite=True)
    run_task("load", path=path)
    run_task("finalise", path=path)

    dataset = datasets.open_dataset(path)
    assert dataset.variables == ["TEMPERATURE_90"]
    assert dataset.shape == (2, 1, 1, 29 * 39)
    assert tuple(dataset.field_shape) == (29, 39)
    assert dataset.resolution == "lambert"

    for i, name in enumerate(("VGeometry_hybridP", "forecast")):
        field = temperature(name)
        # the dataset holds single precision
        assert np.array_equal(dataset[i, 0, 0], field.to_numpy().astype(np.float32).ravel())

    latitudes, longitudes = field.geography.latlons()
    assert np.array_equal(dataset.latitudes, latitudes.ravel())
    assert np.array_equal(dataset.longitudes, longitudes.ravel())


def test_accumulate(dated):
    # the 4 h before 02 UTC are the record valid at 00 UTC plus the one valid at 02 UTC
    (field,) = select(accumulate(END, END, "4h", {"path": dated, "param": "TEMPERATURE", "levelist": 90}))

    expected = temperature("VGeometry_hybridP").to_numpy() + temperature("forecast").to_numpy()
    assert np.array_equal(field.to_numpy().ravel(), expected.ravel())
    assert field.get("time.valid_datetime") == END
    assert field.get("labels.name") == "TEMPERATURE_90"
    assert process(field) == ("accumulation", ["0h", "4h"])


def test_accumulate_one_record(dated):
    (field,) = select(accumulate(END, END, "2h", {"path": dated, "param": "TEMPERATURE", "levelist": 90}))

    assert np.array_equal(field.to_numpy().ravel(), temperature("forecast").to_numpy().ravel())
    assert process(field) == ("accumulation", ["0h", "2h"])


def test_accumulate_missing_file(dated):
    # the 6 h before 02 UTC start with the record valid at 22 UTC the day before
    with pytest.raises(FileNotFoundError, match="fa_2017021322"):
        select(accumulate(END, END, "6h", {"path": dated, "param": "TEMPERATURE", "levelist": 90}))


@pytest.fixture
def source(dated):
    from earthkit_data_fa.anemoi_source import FASource

    context = types.SimpleNamespace(trace=lambda *args: None)
    return FASource(context, dated, param="TEMPERATURE", levelist=90)


def test_interval_asked_twice(source, decode_calls):
    from anemoi.datasets.create.arguments import Intervals
    from anemoi.datasets.create.intervals import SignedInterval

    # two dates that share an interval both ask for it
    interval = SignedInterval(START, END)
    fields = source.execute_intervals(Intervals([END], [interval, interval]))

    assert len(fields) == 1
    assert fields[0].metadata(("date", "time", "startStep", "endStep")) == (20170214, 0, 0, 2)
    assert fields[0].metadata(("validityDate", "validityTime")) == (20170214, 200)
    assert fields[0].get(collections="metadata.mars") == {"param": "TEMPERATURE", "levtype": "hybrid", "levelist": 90}
    # what is not about the interval is still the metadata of the record
    assert fields[0].metadata("name") == "S090TEMPERATURE"
    assert decode_calls == []


def test_interval_with_base_time(source):
    from anemoi.datasets.create.arguments import Intervals
    from anemoi.datasets.create.intervals import SignedInterval

    with pytest.raises(ValueError, match="base time"):
        source.execute_intervals(Intervals([END], [SignedInterval(START, END, base=START)]))
