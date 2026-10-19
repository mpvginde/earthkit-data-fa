import datetime
import pickle

import earthkit.data as ekd
import numpy as np
import pytest
from conftest import sample

from earthkit_data_fa.field import _MODEL_NAMES, parse_name


@pytest.fixture(scope="module")
def forecast():
    return ekd.from_source("fa", sample("forecast")).to_fieldlist()


@pytest.fixture(scope="module")
def analysis():
    return ekd.from_source("fa", sample("analysis")).to_fieldlist()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("S087TEMPERATURE", ("TEMPERATURE", 87, "hybrid")),
        ("S001WIND.U.PHYS", ("WIND.U.PHYS", 1, "hybrid")),
        ("P85000TEMPERATUR", ("TEMPERATURE", 850, "pressure")),
        ("P00000GEOPOTENTI", ("GEOPOTENTIEL", 1000, "pressure")),
        ("P00050TEMPERATUR", ("TEMPERATURE", 0.5, "pressure")),
        ("H00002TEMPERATUR", ("TEMPERATURE", 2, "height_above_ground_level")),
        # cut by the level prefix, or to 12 characters by the model: restored to the full name
        ("P85000VENT_MERID", ("VENT_MERIDIEN", 850, "pressure")),
        ("P85000HUMI.SPECI", ("HUMI.SPECIFIQUE", 850, "pressure")),
        ("P50000RAD_SOLID_", ("RAD_SOLID_WATER", 500, "pressure")),
        ("KT273HUMI_RELATI", ("HUMI_RELATIVE", 273, "temperature")),
        ("S087TOTAL_TUR_EN", ("TOTAL_TUR_ENERGY", 87, "hybrid")),
        ("S087SHEAR TERM T", ("SHEAR TERM TURB", 87, "hybrid")),
        ("S087HUMI.SPECIFI", ("HUMI.SPECIFIQUE", 87, "hybrid")),
        ("S050VENT_MERIDIE", ("VENT_MERIDIEN", 50, "hybrid")),
        ("V015POT_VORTICIT", ("POT_VORTICITY", 15, "potential_vorticity")),
        ("S087FONC.COURANT", ("FONC.COURANT", 87, "hybrid")),
        # fills the 16 characters without being cut, or not a known name: left alone
        ("H00010VENT_ZONAL", ("VENT_ZONAL", 10, "height_above_ground_level")),
        ("P85000DIVERGENCE", ("DIVERGENCE", 850, "pressure")),
        ("S001UNKNOWN_FIEL", ("UNKNOWN_FIEL", 1, "hybrid")),
        ("P85000TEMP", ("TEMP", 850, "pressure")),
        ("V015TEMPERATURE", ("TEMPERATURE", 15, "potential_vorticity")),
        ("T300VENT_ZONAL", ("VENT_ZONAL", 300, "potential_temperature")),
        ("KT273ISOT_ALTIT", ("ISOT_ALTIT", 273, "temperature")),
        ("SURFTEMPERATURE", ("SURFTEMPERATURE", None, "surface")),
        ("MSLPRESSURE", ("MSLPRESSURE", None, "mean_sea")),
        ("CLSTEMPERATURE", ("CLSTEMPERATURE", None, None)),
        ("SPECSURFGEOPOTEN", ("SPECSURFGEOPOTEN", None, None)),
        ("TOPMESO ENTH", ("TOPMESO ENTH", None, None)),
        ("PROFTEMPERATURE", ("PROFTEMPERATURE", None, None)),
        ("SUNSHINE.DUR", ("SUNSHINE.DUR", None, None)),
    ],
)
def test_parse_name(name, expected):
    assert parse_name(name) == expected


@pytest.mark.parametrize("length", [10, 11, 12])
def test_cut_names_are_unambiguous(length):
    # a level prefix leaves 10, 11 or 12 characters for the field name
    cut = [name[:length] for name in _MODEL_NAMES if len(name) > length]
    assert len(cut) == len(set(cut))


def test_restored_names_in_file(analysis):
    restored = analysis.sel({"parameter.variable": "TOTAL_TUR_ENERGY"})
    assert len(restored) == 87
    assert restored[0].get("metadata.name").endswith("TOTAL_TUR_EN")
    assert len(analysis.sel({"parameter.variable": "SHEAR TERM TURB"})) == 87
    assert len(analysis.sel({"parameter.variable": "TOTAL_TUR_EN"})) == 0


def test_from_source(forecast):
    data = ekd.from_source("fa", sample("forecast"))
    assert "fieldlist" in data.available_types and "xarray" in data.available_types
    assert len(forecast) == 134
    assert len(forecast.ls()) == 134


def test_field_metadata(forecast):
    selected = forecast.sel({"parameter.variable": "TEMPERATURE", "vertical.level": 90})
    assert len(selected) == 1
    field = selected[0]

    assert field.shape == (29, 39)
    assert field.get("vertical.level_type") == "hybrid"
    assert field.get("time.base_datetime") == datetime.datetime(2017, 2, 14)
    assert field.get("time.valid_datetime") == datetime.datetime(2017, 2, 14, 2)
    assert field.get("time.step") == datetime.timedelta(hours=2)
    assert field.get("geography.grid_type") == "lambert"

    a, b = field.vertical.coefficients()
    assert len(a) == len(b) == 91
    assert field.vertical.number_of_levels() == 90

    assert field.get("metadata.name") == "S090TEMPERATURE"
    assert field.get("metadata.spectral") is True
    assert field.get("metadata.dx") == 2500.0
    assert field.metadata("proj_string") == field.geography.projection().to_proj_string()
    assert field.get("metadata.no_such_key", default="x") == "x"
    with pytest.raises(KeyError):
        field.metadata("no_such_key")


def test_frame_metadata_fallthrough(forecast):
    # every Frame field (fafile.py) is reachable by name, with no separate list to maintain
    field = forecast.sel({"metadata.name": "S090TEMPERATURE"})[0]

    assert field.metadata("nsmax") == 19
    assert field.metadata("nmsmax") == 24
    assert (field.metadata("ndlun"), field.metadata("ndlux")) == (1, 39)
    assert field.metadata("lon0") == pytest.approx(-51.5)
    assert field.metadata("nlev") == 90

    a, b = field.metadata("a"), field.metadata("b")
    assert (len(a), len(b)) == (91, 91)
    assert (a, b) == field.vertical.coefficients()

    # the record itself still takes precedence over any same-named frame attribute
    assert field.metadata("spectral") is True
    # validity info is reached the same way
    assert field.metadata("time_range_indicator") == 10


def test_grib_grid_keys(forecast):
    # the two GRIB keys anemoi-datasets reads to describe the resolution of a dataset
    assert forecast[0].metadata("gridType") == "lambert"
    assert forecast[0].get("metadata.iDirectionIncrementInDegrees", default=None) is None

    regular = ekd.from_source("fa", sample("regLL_small")).to_fieldlist()[0]
    assert regular.metadata("gridType") == "regular_ll"
    assert regular.metadata("iDirectionIncrementInDegrees") == pytest.approx(regular.metadata("dx"))


def test_selection(forecast):
    assert len(forecast.sel({"metadata.spectral": True})) == 91
    assert len(forecast.sel({"vertical.level_type": "hybrid"})) == 90
    assert len(forecast.sel({"parameter.variable": "TEMPERATURE"})) == 90
    surface = forecast.sel({"metadata.name": "SURFTEMPERATURE"})[0]
    assert surface.get("vertical.level_type") == "surface"
    assert surface.get("vertical.level") is None


def test_values(forecast):
    spectral = forecast.sel({"metadata.name": "S090TEMPERATURE"})[0]
    gridpoint = forecast.sel({"metadata.name": "SURFTEMPERATURE"})[0]

    values = spectral.to_numpy()
    assert values.shape == (29, 39) and values.dtype == np.float64
    assert spectral.to_numpy(flatten=True).shape == (29 * 39,)
    assert spectral.to_numpy(dtype=np.float32).dtype == np.float32
    assert spectral.values.shape == (29 * 39,)
    # lowest model level and surface temperature are close
    assert np.abs(values - gridpoint.to_numpy()).mean() < 2.0

    lat, lon = spectral.geography.latlons()
    assert lat.shape == lon.shape == (29, 39)
    assert (lat[0, 0], lon[0, 0]) == pytest.approx((5.4310, -54.6278), abs=1e-4)


def test_spectral_surface_pressure_is_in_pascal(forecast):
    field = forecast.sel({"metadata.name": "SURFPRESSION"})[0]
    assert field.get("metadata.spectral") is True

    pressure = field.to_numpy()
    assert 85000 < pressure.min() < pressure.max() < 106000
    assert np.array_equal(pickle.loads(pickle.dumps(field)).to_numpy(), pressure)


def test_nothing_is_decoded_until_values_are_used(decode_calls):
    fieldlist = ekd.from_source("fa", sample("forecast")).to_fieldlist()
    fieldlist.ls()
    selected = fieldlist.sel({"parameter.variable": "TEMPERATURE"}).order_by("vertical.level")
    assert selected[0].shape == (29, 39)
    dataset = selected.to_xarray()
    assert decode_calls == []

    assert dict(dataset.sizes) == {"level": 90, "y": 29, "x": 39}
    assert set(dataset.data_vars) == {"TEMPERATURE"}
    assert dataset["latitude"].shape == (29, 39)

    level = dataset["TEMPERATURE"].sel(level=90).values
    assert len(decode_calls) == 1
    assert np.array_equal(level, selected.sel({"vertical.level": 90})[0].to_numpy())


def test_pickle(forecast):
    for name in ("S090TEMPERATURE", "SURFTEMPERATURE"):
        field = forecast.sel({"metadata.name": name})[0]
        clone = pickle.loads(pickle.dumps(field))

        assert clone.get("metadata.name") == name
        assert clone.get("vertical.level") == field.get("vertical.level")
        assert clone.get("vertical.level_type") == field.get("vertical.level_type")
        assert clone.get("time.valid_datetime") == field.get("time.valid_datetime")
        assert np.array_equal(clone.to_numpy(), field.to_numpy())


def test_several_files():
    # the second file holds one field of the same run, at the initial time
    fieldlist = ekd.from_source("fa", [sample("forecast"), sample("VGeometry_hybridP")]).to_fieldlist()
    assert len(fieldlist) == 135

    selected = fieldlist.sel({"metadata.name": "S090TEMPERATURE"})
    assert [f.get("time.step") for f in selected] == [datetime.timedelta(hours=2), datetime.timedelta(0)]
    assert len({f.get("time.base_datetime") for f in selected}) == 1


def test_file_without_valid_date():
    fieldlist = ekd.from_source("fa", sample("lambert_HN")).to_fieldlist()
    assert len(fieldlist) == 3
    assert fieldlist[0].to_numpy().shape == (300, 300)


def test_unsupported_files():
    with pytest.raises(ValueError):
        ekd.from_source("fa", sample("lfi"))
    with pytest.raises(NotImplementedError):
        ekd.from_source("fa", sample("gaussC1"))
