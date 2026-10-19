"""anemoi-datasets source for FA files: ``input: fa: {path: ...}`` in a recipe."""

import glob
import logging

import earthkit.data as ekd
from anemoi.datasets.create.intervals import timedelta_to_step
from anemoi.datasets.create.source import Source
from anemoi.transform import FieldList
from anemoi.transform.fields import metadata_key
from earthkit.data.core.field import Field
from earthkit.data.utils.patterns import Pattern

__all__ = ["FASource"]

LOG = logging.getLogger(__name__)


class _IntervalMetadata:
    """Raw metadata of a field presented as an accumulation over an interval.

    The ``accumulate`` source of anemoi-datasets reads the interval of a field from GRIB keys,
    and tells variables apart by the ``mars`` namespace. An FA record has neither, so both are
    made here, from the interval the field was asked for. Any other key goes to the FA metadata.
    """

    def __init__(self, field, interval):
        self._metadata = field._get_private_data("metadata")
        start, end = interval.min, interval.max
        self._keys = {
            "date": int(f"{start:%Y%m%d}"),
            "time": int(f"{start:%H%M}"),
            "startStep": 0,
            "endStep": timedelta_to_step(end - start),
            "validityDate": int(f"{end:%Y%m%d}"),
            "validityTime": int(f"{end:%H%M}"),
        }
        mars = {"param": "parameter.variable", "levtype": "vertical.level_type", "levelist": "vertical.level"}
        mars = {k: field.get(v, default=None) for k, v in mars.items()}
        self._mars = {k: v for k, v in mars.items() if v is not None}

    def metadata(self, key, default=None, *, astype=None, raise_on_missing=False):
        if key in self._keys:
            value = self._keys[key]
            return astype(value) if callable(astype) else value
        return self._metadata.metadata(key, default, astype=astype, raise_on_missing=raise_on_missing)

    def as_namespace(self, name):
        return dict(self._mars) if name == "mars" else {}


class FASource(Source):
    """Limited-area FA files as an anemoi-datasets source.

    Parameters
    ----------
    context:
        The anemoi-datasets context the source is created in.
    path: str or list of str
        The FA file(s). A path may hold glob wildcards and ``{date:strftime(...)}``, which is
        replaced by each of the dates being built.
    **kwargs:
        Which fields to keep, as for ``FieldList.sel()``. A key is an earthkit-data one
        (``parameter.variable``, ``vertical.level``, ``vertical.level_type``), its legacy name
        (``param``, ``levelist``, ``levtype``) or raw FA metadata (``metadata.name``; a bare
        ``name`` is taken by the anemoi registry).
    """

    def __init__(self, context, path, **kwargs):
        super().__init__(context)
        self.paths = [path] if isinstance(path, str) else list(path)
        self.kwargs = kwargs

    def execute_valid_dates(self, dates):
        fields = self._read(dates)
        if not fields:
            dates = [d.isoformat() for d in dates]
            LOG.warning(f"No fields found for {dates} in {self.paths} (kwargs={self.kwargs})")
        return FieldList.from_fields(fields)

    def execute_intervals(self, intervals):
        """Return what the ``accumulate`` source asks for: the fields valid at the end of each interval.

        An FA file does not say over which period a record is accumulated. It is taken to be
        the interval, which comes from the ``covering`` given in the recipe.
        """
        fields = []
        done = set()
        for interval in intervals.intervals:
            if interval.base is not None:
                raise ValueError(
                    f"FA files are found by valid time, not by base time: {interval}. "
                    "Use '*' as the base time in the 'covering' of 'accumulate'."
                )
            # an interval is asked once for each of the dates it contributes to
            if (interval.min, interval.max) in done:
                continue
            done.add((interval.min, interval.max))

            for field in self._read([interval.max]):
                accumulated = Field.from_field(field)
                accumulated._set_private_data("metadata", _IntervalMetadata(field, interval))
                fields.append(accumulated)

        return FieldList.from_fields(fields)

    def _read(self, dates):
        """Return the selected fields that are valid at ``dates``; nothing is decoded."""
        dates = [d.isoformat() for d in dates]

        selection = dict(self.kwargs)
        remapping = {k: "{" + metadata_key(k, default=f"metadata.{k}") + "}" for k in self.kwargs}
        if dates:
            selection["valid_datetime"] = dates
            remapping["valid_datetime"] = "{time.valid_datetime}"

        fields = []
        for pattern in self.paths:
            paths = Pattern(pattern).substitute(date=dates) if "{" in pattern else [pattern]
            for path in paths:
                # a path that matches nothing is kept, so that the missing file is reported
                for p in sorted(glob.glob(path)) or [path]:
                    self.context.trace(self.emoji, "PATH", p)
                    fields.extend(ekd.from_source("fa", p).to_fieldlist().sel(**selection, remapping=remapping))
        return fields
