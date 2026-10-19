import math
import pickle

import numpy as np
import pytest
from conftest import sample

from earthkit_data_fa.fafile import FAFile
from earthkit_data_fa.geography import FAGeography

# (longitude, latitude) of grid points (0, 0) and (i=24, j=36), from EPyGrAM's test_geometry_methods.py
EPYGRAM_POINTS = {
    "lambert_HS": ((-88.772039618271094, -60.859390190567119), (-84.95623428711568, -58.475140869628035)),
    "mercator_HN": ((-81.51099961956092, -6.323778282067483), (-78.05772877762075, -1.1566717493466303)),
    "stereopol_HN": ((-53.423814350311183, 42.887302601956691), (-51.28805020370288, 48.00255024785442)),
    "regLL_small": ((-8.0, 38.0), (-7.4, 38.9)),
}

GRID_TYPES = {
    "clim": "lambert",
    "forecast": "lambert",
    "lambert_HN": "lambert",
    "lambert_HS": "lambert",
    "mercator_HN": "mercator",
    "mercator_HS": "mercator",
    "stereopol_HN": "polar_stereographic",
    "stereopol_HS": "polar_stereographic",
    "regLL_small": "regular_ll",
    "regLL_large": "regular_ll",
}


def wrap(degrees):
    return (degrees + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("name,grid_type", GRID_TYPES.items())
def test_corners_match_frame(name, grid_type):
    f = FAFile(sample(name))
    geography = FAGeography(f.frame)
    assert geography.grid_type() == grid_type

    # the frame stores the SW and NE corners of the C+I zone independently of the centre
    sinlat = f._article(f._read_index(), "CADRE-SINLATITUD", ">f8")
    lon_sw, lat_sw, lon_ne, lat_ne = (math.degrees(v) for v in sinlat[12:16])

    lon, lat = geography.longitudes(), geography.latitudes()
    assert lon.shape == lat.shape == geography.shape() == (f.frame.ny, f.frame.nx)
    assert wrap(lon[0, 0] - lon_sw) == pytest.approx(0, abs=1e-9)
    assert wrap(lon[-1, -1] - lon_ne) == pytest.approx(0, abs=1e-9)
    assert lat[0, 0] == pytest.approx(lat_sw, abs=1e-9)
    assert lat[-1, -1] == pytest.approx(lat_ne, abs=1e-9)


@pytest.mark.parametrize("name,points", EPYGRAM_POINTS.items())
def test_points_match_epygram(name, points):
    geography = FAGeography(FAFile(sample(name)).frame)
    lon, lat = geography.longitudes(), geography.latitudes()

    assert (lon[0, 0], lat[0, 0]) == pytest.approx(points[0], abs=1e-9)
    assert (lon[36, 24], lat[36, 24]) == pytest.approx(points[1], abs=1e-9)


def test_lambert_projection():
    f = FAFile(sample("forecast"))
    geography = FAGeography(f.frame)
    projection = geography.projection()

    assert projection.to_proj_string() == (
        "+proj=lcc +lat_1=5 +lat_2=5 +lat_0=5 +lon_0=-51.5 +units=m +a=6371229.0 +b=6371229.0 +no_defs"
    )
    assert type(projection).__name__ == "LambertConformal"
    assert projection.parameters["standard_parallels"] == (5.0, 5.0)
    assert projection.globe == {"semimajor_axis": 6371229.0, "semiminor_axis": 6371229.0}

    x, y = geography.x(), geography.y()
    assert x.shape == y.shape == (29, 39)
    assert np.allclose(np.diff(x[0]), 2500.0) and np.allclose(np.diff(y[:, 0]), 2500.0)
    # the domain centre is 298.75 km west and 83.75 km north of the projection origin
    assert x.mean() == pytest.approx(-298750, abs=1e-3) and y.mean() == pytest.approx(83750, abs=1e-3)
    assert geography.distinct_latitudes() is None

    north, west, south, east = geography.area()
    assert (south, north) == pytest.approx((5.43, 6.06), abs=0.01)
    assert (west, east) == pytest.approx((-54.63, -53.77), abs=0.01)
    assert geography.bounding_box().north == pytest.approx(north)


@pytest.mark.parametrize(
    "name,cls,proj", [("mercator_HN", "Mercator", "merc"), ("stereopol_HS", "PolarStereographic", "stere")]
)
def test_other_projections(name, cls, proj):
    geography = FAGeography(FAFile(sample(name)).frame)
    projection = geography.projection()

    assert type(projection).__name__ == cls
    assert f"+proj={proj} " in projection.to_proj_string()
    assert geography.bounding_box() is not None


def test_regular_latlon():
    geography = FAGeography(FAFile(sample("regLL_small")).frame)

    lat, lon = geography.distinct_latitudes(), geography.distinct_longitudes()
    assert (lat[0], lat[-1], len(lat)) == pytest.approx((38.0, 53.0, 601))
    assert (lon[0], lon[-1], len(lon)) == pytest.approx((-8.0, 12.0, 801))
    assert np.array_equal(geography.latitudes()[:, 0], lat)
    assert np.array_equal(geography.x(), geography.longitudes())
    assert "+proj=longlat" in geography.projection().to_proj_string()


def test_grid_id_and_pickle():
    forecast = FAGeography(FAFile(sample("forecast")).frame)
    same_domain = FAGeography(FAFile(sample("VGeometry_hybridP")).frame)
    other = FAGeography(FAFile(sample("lambert_HN")).frame)

    assert forecast.unique_grid_id() == same_domain.unique_grid_id()
    assert forecast.unique_grid_id() != other.unique_grid_id()

    clone = pickle.loads(pickle.dumps(other))
    assert clone.unique_grid_id() == other.unique_grid_id()
    assert np.array_equal(clone.latitudes(), other.latitudes())
