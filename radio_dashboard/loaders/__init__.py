"""Per-instrument loader registry.

The registry is populated lazily: a module is imported only when its key
is first requested, so a missing optional dependency (e.g. sigpyproc for
I-LOFAR) does not prevent the rest of the dashboard from starting.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any
from ._common import DynamicSpectrum

LOADER_MODULES = {
    "swaves":  "radio_dashboard.loaders.swaves",
    "wind":    "radio_dashboard.loaders.wind",
    "psp":     "radio_dashboard.loaders.psp",
    "solo":    "radio_dashboard.loaders.solo",
    "nda":     "radio_dashboard.loaders.nda",
    "orfees":  "radio_dashboard.loaders.orfees",
    "lofar":   "radio_dashboard.loaders.lofar",
    "ilofar":  "radio_dashboard.loaders.ilofar",
    "nenufar": "radio_dashboard.loaders.nenufar",
    "ovsa":    "radio_dashboard.loaders.ovsa",
}


def load(instrument_key: str, **kwargs: Any) -> DynamicSpectrum:
    """Dispatch to the right loader. All instrument-specific kwargs pass through."""
    mod_path = LOADER_MODULES.get(instrument_key)
    if mod_path is None:
        raise KeyError(f"Unknown instrument key: {instrument_key!r}")
    mod = import_module(mod_path)
    return mod.load(**kwargs)


__all__ = ["DynamicSpectrum", "load", "LOADER_MODULES"]
