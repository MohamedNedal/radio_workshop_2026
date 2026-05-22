"""Defaults and instrument registry for the radio-dashboard tool.

Single source of truth for: where data lives on disk, which instruments
exist, what bands/products they expose, and which of them are space-based
(so light-travel-time correction is meaningful).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# Default cache root. Override at runtime via the sidebar or env var.
DEFAULT_DATA_ROOT = os.environ.get(
    "RADIO_DASHBOARD_DATA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_data")),
)

DEFAULT_OUTPUTS_DIR = os.environ.get(
    "RADIO_DASHBOARD_OUTPUTS",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs")),
)


@dataclass(frozen=True)
class Instrument:
    """Static metadata for one instrument."""

    key: str                       # short identifier used in the sidebar and file paths
    label: str                     # human-readable name shown in the UI
    platform: str                  # "ground" or "space"
    freq_min_mhz: float
    freq_max_mhz: float
    levels: tuple[str, ...] = ()   # supported processing levels
    versions: tuple[str, ...] = () # known file versions (most recent first)
    downloadable: bool = False     # True if downloaders.py can fetch from a public archive
    sc_name: Optional[str] = None  # spacecraft name for SPICE / sunpy-coordinates queries
    notes: str = ""


INSTRUMENTS: dict[str, Instrument] = {
    "swaves": Instrument(
        key="swaves",
        label="STEREO-A / SWAVES",
        platform="space",
        freq_min_mhz=0.0026,
        freq_max_mhz=16.025,
        levels=("l2",),
        versions=("v02", "v01"),
        downloadable=True,
        sc_name="STEREO-A",
        notes="Level-2 combined CDF on SPDF. Uses avg_intens_ahead by default.",
    ),
    "wind": Instrument(
        key="wind",
        label="Wind / WAVES",
        platform="space",
        freq_min_mhz=0.004,
        freq_max_mhz=13.825,
        levels=("l2", "h1"),
        # Versions are not user-selectable: the downloader scans the SPDF
        # directory listing and picks the highest version available.
        versions=(),
        downloadable=True,
        sc_name="Wind",
        notes="TNR + RAD1 + RAD2 combined onto a log frequency axis.",
    ),
    "psp": Instrument(
        key="psp",
        label="Parker Solar Probe / FIELDS RFS",
        platform="space",
        freq_min_mhz=0.0105,
        freq_max_mhz=19.2,
        levels=("l2", "l3"),
        versions=("v03", "v02"),
        downloadable=True,
        sc_name="Parker Solar Probe",
        notes="LFR + HFR stitched on a shared time axis with overlap deduplicated.",
    ),
    "solo": Instrument(
        key="solo",
        label="Solar Orbiter / RPW (HFR+TNR)",
        platform="space",
        freq_min_mhz=0.004,
        freq_max_mhz=16.0,
        levels=("l2", "l3"),
        versions=(),
        downloadable=True,
        sc_name="Solar Orbiter",
        notes="Optional: pulled via sunpy.net.Fido (cdaweb provider). Disabled if Fido fails.",
    ),
    "nda": Instrument(
        key="nda",
        label="Nancay Decameter Array",
        platform="ground",
        freq_min_mhz=10.0,
        freq_max_mhz=80.0,
        # "Level" here is overloaded to channel selection (LH, RH, RH-LH).
        levels=("ch1", "ch2", "diff"),
        versions=(),
        downloadable=True,
        notes="ch1 = LH circular, ch2 = RH circular, diff = RH - LH.",
    ),
    "orfees": Instrument(
        key="orfees",
        label="ORFEES (Nancay)",
        platform="ground",
        freq_min_mhz=144.0,
        freq_max_mhz=1004.0,
        levels=(),
        versions=(),
        downloadable=False,
        notes="Five sub-bands B1-B5 stacked; 1 s resample by default.",
    ),
    "lofar": Instrument(
        key="lofar",
        label="LOFAR (core / international)",
        platform="ground",
        freq_min_mhz=10.0,
        freq_max_mhz=240.0,
        levels=("LBA", "HBA"),
        versions=(),
        downloadable=False,
        notes="FITS blocks: HDU[0] data, HDU[1] FREQ, HDU[2] TIME.",
    ),
    "ilofar": Instrument(
        key="ilofar",
        label="I-LOFAR / REALTA",
        platform="ground",
        freq_min_mhz=10.0,
        freq_max_mhz=270.0,
        levels=("I", "V"),
        versions=(),
        downloadable=False,
        notes="sigproc filterbank. Modes 3 + 5 + 7 with NaN-filled gaps.",
    ),
    "nenufar": Instrument(
        key="nenufar",
        label="NenuFAR",
        platform="ground",
        freq_min_mhz=10.0,
        freq_max_mhz=85.0,
        levels=("stokesI", "stokesV_over_I"),
        versions=(),
        downloadable=False,
        notes="Per-burst pickle exports from the NenuFAR pipeline.",
    ),
    "ovsa": Instrument(
        key="ovsa",
        label="OVRO-LWA / EOVSA",
        platform="ground",
        freq_min_mhz=20.0,
        freq_max_mhz=18000.0,
        levels=(),
        versions=(),
        downloadable=False,
        notes="FITS calibrated solar spectrum in SFU.",
    ),
}


# Display order in the sidebar: lowest frequency at the top of the figure
# is achieved by plotting space-based panels first.
DISPLAY_ORDER = (
    "swaves", "wind", "psp", "solo",
    "nda", "nenufar", "lofar", "ilofar", "orfees", "ovsa",
)


def data_dir_for(key: str, root: str = DEFAULT_DATA_ROOT) -> str:
    """Return the on-disk cache directory for a given instrument key."""
    return os.path.join(root, key)
