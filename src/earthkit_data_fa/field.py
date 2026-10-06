"""Build lazy earthkit Fields from the records of an FA file."""

import numpy as np
from earthkit.data.core.field import Field
from earthkit.data.field.component.vertical import ParametricVertical, Vertical
from earthkit.data.field.handler.data import DataFieldComponentHandler
from earthkit.data.field.handler.geography import GeographyFieldComponentHandler
from earthkit.data.field.handler.parameter import ParameterFieldComponentHandler
from earthkit.data.field.handler.time import TimeFieldComponentHandler
from earthkit.data.field.handler.vertical import VerticalFieldComponentHandler

from . import decode as _decode
from .geography import FAGeography, proj_string

# name prefixes followed by a level value: (number of digits, earthkit level type)
_NUMBERED_LEVELS = {
    "S": (3, "hybrid"),
    "P": (5, "pressure"),
    "H": (5, "height_above_ground_level"),
    "V": (3, "potential_vorticity"),
    "T": (3, "potential_temperature"),
    "KT": (3, "temperature"),
}
_SINGLE_LEVELS = {"SURF": "surface", "MSL": "mean_sea", "MER": "mean_sea"}

# Upper-air field names (defaults of arpifs/setup/sufa.F90 and suafn1.F90, cycle 46), longer
# than the 10 characters a level prefix always leaves. Where the model itself defines a name
# cut to 12 characters (VENT_MERIDIE, HUMI.SPECIFI), the last word is spelled out here.
_MODEL_NAMES = (
    "ABS_VORTICITY",
    "CLOUD_FRACTION",
    "CLOUD_WATER",
    "CV_PREC_FLUX",
    "DD_MESH_FRAC",
    "FONC.COURANT",
    "GEOPOTENTIEL",
    "HUMI.SPECIFIQUE",
    "HUMID_DENSITY",
    "HUMI_RELATIVE",
    "ICE_CRYSTAL",
    "ICINGX.AERO",
    "LIQUID_WATER",
    "MIXING_LENGTH",
    "MONTGOMERY GEOPOTENTIAL",
    "POT.VITESSE",
    "POT_VORTICITY",
    "PRESS.DEPARTURE",
    "PSHI_CONV_CLOUD",
    "Q SRC TERM TURB",
    "QVA_AMPLITUDE",
    "RAD_LIQUID_WATER",
    "RAD_SOLID_WATER",
    "RK_TEMPTEND",
    "S SRC TERM TURB",
    "SHEAR TERM TURB",
    "SHEAR_DEFORMATION",
    "SIM_REFLECTIVITY",
    "SOLID_WATER",
    "ST_PREC_FLUX",
    "STRET_DEFORMATION",
    "TEMPE_POTENTIELLE",
    "TEMPERATURE",
    "THETAPWP_FLUX",
    "THETA_EQUIVALENT",
    "THETA_PRIM_W",
    "THETA_VIRTUAL",
    "TOTAL_TUR_ENERGY",
    "UD_ENTRAINMENT",
    "UD_MESH_FRAC",
    "VENT_MERIDIEN",
    "VERT.VELOCITY",
    "VERTIC.DIVERGENCE",
    "VIRT_P_TEMPERATURE",
    "VITESSE_VERTICALE",
    "WIND.U.PHYS",
    "WIND.V.PHYS",
    "WIND_VELOCITY",
)


def _restore(variable, max_length):
    """Return the full name of a variable cut to fit, with its level prefix, in 16 characters."""
    if len(variable) < max_length:
        return variable
    matches = [n for n in _MODEL_NAMES if len(n) > max_length and n.startswith(variable)]
    return matches[0] if len(matches) == 1 else variable


def parse_name(name):
    """Split an FA field name into (variable, level, level type)."""
    for prefix, (digits, level_type) in _NUMBERED_LEVELS.items():
        end = len(prefix) + digits
        if name.startswith(prefix) and name[len(prefix) : end].isdigit() and len(name) > end:
            level = int(name[len(prefix) : end])
            if level_type == "pressure":
                # stored in Pa on 5 digits, so 100000 Pa is written as 00000
                level = level or 100000
                level = level // 100 if level % 100 == 0 else level / 100
            return _restore(name[end:], 16 - end), level, level_type
    for prefix, level_type in _SINGLE_LEVELS.items():
        if name.startswith(prefix):
            return name, None, level_type
    return name, None, None


class FAData(DataFieldComponentHandler):
    def __init__(self, fafile, record):
        self._file = fafile
        self._record = record

    def get_values(self, dtype=None, copy=True):
        values = _decode.decode(self._file.read(self._record), self._file.frame)
        if self._record.spectral and self._record.name == "SURFPRESSION":
            # in spectral space the model stores the logarithm of surface pressure
            np.exp(values, out=values)
        return values if dtype is None else values.astype(dtype, copy=False)

    def __getstate__(self):
        return {"file": self._file, "record": self._record}

    def __setstate__(self, state):
        self.__init__(state["file"], state["record"])


class _Vertical(Vertical):
    # earthkit-data's own state methods use mismatched keys, so fields would not unpickle
    def __getstate__(self):
        return {"level": self._level, "level_type": self._type.name}

    def __setstate__(self, state):
        self.__init__(**state)


class _HybridVertical(ParametricVertical):
    def __getstate__(self):
        return {"level": self._level, "level_type": self._type.name, "coefficients": self.coefficients()}

    def __setstate__(self, state):
        self.__init__(**state)


class FAMetadata:
    """Raw FA metadata of one field, reached as ``field.get("metadata.<key>")``.

    Per-record keys (``name``, ``spectral``, ``ngrib``, ``nbits``) take precedence; anything
    else falls through to the file's validity info and then to the ``Frame`` dataclass
    (``fafile.py``), so every frame attribute (``nsmax``, ``ndlun``, ``lon0``, ``a``, ...) is
    reached under its own name, with no separate list to keep in sync. ``proj_string`` is the
    one computed addition, matching ``geography.projection().to_proj_string()``.
    """

    def __init__(self, record, frame, validity):
        self._record = record
        self._frame = frame
        self._validity = validity

    def metadata(self, key, default=None, *, astype=None, raise_on_missing=False):
        if key in ("name", "spectral", "ngrib", "nbits"):
            value = getattr(self._record, key)
        elif key == "proj_string":
            value = proj_string(self._frame)
        elif key in self._validity:
            value = self._validity[key]
        elif hasattr(self._frame, key):
            value = getattr(self._frame, key)
        elif raise_on_missing:
            raise KeyError(key)
        else:
            return default
        return astype(value) if callable(astype) and value is not None else value


def make_fields(fafile):
    """Return one Field per field record; no values are read."""
    frame = fafile.frame
    geography = GeographyFieldComponentHandler.from_component(FAGeography(frame))
    time = None
    if fafile.base_datetime is not None:
        time = TimeFieldComponentHandler.from_dict({"base_datetime": fafile.base_datetime, "step": fafile.step})
    coefficients = (frame.a, frame.b)
    verticals = {}

    fields = []
    for record in fafile.records:
        variable, level, level_type = parse_name(record.name)

        key = (level, level_type)
        if key not in verticals:
            if level_type == "hybrid":
                vertical = _HybridVertical(level=level, level_type=level_type, coefficients=coefficients)
            elif level_type:
                vertical = _Vertical(level=level, level_type=level_type)
            else:
                vertical = None
            verticals[key] = vertical and VerticalFieldComponentHandler.from_component(vertical)

        field = Field(
            data=FAData(fafile, record),
            time=time,
            parameter=ParameterFieldComponentHandler.from_dict({"variable": variable}),
            geography=geography,
            vertical=verticals[key],
        )
        field._set_private_data("metadata", FAMetadata(record, frame, fafile.validity))
        fields.append(field)
    return fields
