"""earthkit-data source plugin for FA files: ``earthkit.data.from_source("fa", path)``."""

import os

from earthkit.data.indexing.simple import SimpleFieldList
from earthkit.data.sources import Source

from .fafile import FAFile
from .field import make_fields

__all__ = ["FASource", "FAFile"]


class FASource(Source):
    """Limited-area FA files as a lazy FieldList.

    Parameters
    ----------
    path: str, os.PathLike or a list of them
        The FA file(s) to read.
    """

    def __init__(self, path):
        super().__init__()
        self._paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)

    def mutate(self):
        fields = []
        for path in self._paths:
            fields.extend(make_fields(FAFile(path)))
        return SimpleFieldList(fields)
