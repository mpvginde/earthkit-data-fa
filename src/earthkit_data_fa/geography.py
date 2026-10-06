"""Geography of FA limited-area fields: CRS and coordinates of the C+I zone."""

import math
from functools import cached_property

import numpy as np
from earthkit.data.field.component.geography import GeographyBase
from earthkit.data.utils.bbox import BoundingBox
from earthkit.data.utils.projections import Projection
from earthkit.data.utils.projections import proj as proj_utils

from .fafile import EARTH_RADIUS


class PolarStereographic(Projection):
    PROJ_NAME = "stere"
    CF_GRID_MAPPING_NAME = "polar_stereographic"
    CARTOPY_CRS = "Stereographic"


def proj_string(frame):
    """Return the PROJ definition of the CRS the grid is regular in."""
    # the sphere is given as +a/+b because earthkit does not map +R to the cartopy globe
    sphere = f"+a={EARTH_RADIUS:.1f} +b={EARTH_RADIUS:.1f} +no_defs"
    lon0, lat0 = f"{frame.lon0:.12g}", f"{frame.lat0:.12g}"
    if frame.projection == "regular_ll":
        return f"+proj=longlat {sphere}"
    if frame.projection == "mercator":
        return f"+proj=merc +lon_0={lon0} +units=m {sphere}"
    if frame.projection == "polar_stereographic":
        pole = math.copysign(90, frame.lat0)
        return f"+proj=stere +lat_0={pole:g} +lon_0={lon0} +lat_ts={pole:g} +units=m {sphere}"
    return f"+proj=lcc +lat_1={lat0} +lat_2={lat0} +lat_0={lat0} +lon_0={lon0} +units=m {sphere}"


def _readonly(array):
    array.flags.writeable = False
    return array


def _as(array, dtype):
    return array if dtype is None else array.astype(dtype)


class FAGeography(GeographyBase):
    def __init__(self, frame):
        self._frame = frame

    @cached_property
    def _proj(self):
        import pyproj

        return pyproj.Proj(proj_string(self._frame))

    @cached_property
    def _axes(self):
        """1-D x and y coordinates in the CRS (metres, or degrees for regular_ll)."""
        f = self._frame
        if f.projection == "regular_ll":
            x0, y0 = f.anchor_lon, f.anchor_lat
        else:
            x0, y0 = self._proj(f.anchor_lon, f.anchor_lat)
        x = x0 + (np.arange(f.nx) - f.anchor_i) * f.dx
        y = y0 + (np.arange(f.ny) - f.anchor_j) * f.dy
        if f.projection == "regular_ll":
            x += (x[0] + 180.0) % 360.0 - 180.0 - x[0]
        return _readonly(x), _readonly(y)

    @cached_property
    def _xy(self):
        x, y = np.meshgrid(*self._axes)
        return _readonly(x), _readonly(y)

    @cached_property
    def _lonlat(self):
        if self._frame.projection == "regular_ll":
            return self._xy
        lon, lat = self._proj(*self._xy, inverse=True)
        return _readonly(lon), _readonly(lat)

    def latitudes(self, dtype=None):
        return _as(self._lonlat[1], dtype)

    def longitudes(self, dtype=None):
        return _as(self._lonlat[0], dtype)

    def distinct_latitudes(self, dtype=None):
        if self._frame.projection == "regular_ll":
            return _as(self._axes[1], dtype)
        return None

    def distinct_longitudes(self, dtype=None):
        if self._frame.projection == "regular_ll":
            return _as(self._axes[0], dtype)
        return None

    def x(self, dtype=None):
        return _as(self._xy[0], dtype)

    def y(self, dtype=None):
        return _as(self._xy[1], dtype)

    def shape(self):
        return (self._frame.ny, self._frame.nx)

    def projection(self):
        definition = proj_string(self._frame)
        if self._frame.projection == "polar_stereographic":
            return PolarStereographic(definition, **proj_utils.to_projection_kwargs(definition))
        return Projection.from_proj_string(definition)

    def area(self):
        lon, lat = self._lonlat
        # longitudes relative to the domain centre, so that the antimeridian is handled
        centre = float(lon[lon.shape[0] // 2, lon.shape[1] // 2])
        relative = (lon - centre + 180.0) % 360.0 - 180.0
        return (float(lat.max()), centre + float(relative.min()), float(lat.min()), centre + float(relative.max()))

    def bounding_box(self):
        north, west, south, east = self.area()
        return BoundingBox(north=north, west=west, south=south, east=east)

    def grid(self):
        return None

    def grid_spec(self):
        return None

    def grid_type(self):
        return self._frame.projection

    def unique_grid_id(self):
        f = self._frame
        return (proj_string(f), f.nx, f.ny, f.dx, f.dy, f.anchor_lon, f.anchor_lat, f.anchor_i, f.anchor_j)

    def __getstate__(self):
        return {"frame": self._frame}

    def __setstate__(self, state):
        self.__init__(state["frame"])
