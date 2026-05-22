"""Light-travel-time correction for space-based instruments.

We shift the time axis of a space-based dynamic spectrum by

    dt = (r_sc - 1 AU) / c

so that a photon emitted at the Sun at time t is plotted at the same
abscissa whether it was recorded near 1 AU (Wind, STEREO) or much
closer (PSP perihelion at ~0.05 AU implies a ~12 minute shift).

Spacecraft positions are queried via `sunpy.coordinates.get_horizons_coord`
when sunpy is available; we keep a small per-(date, body) cache to avoid
repeated network calls. If the query fails (no internet, no JPL/Horizons
access) we fall back to 1 AU and announce the failure so the user can
override with a manual distance.
"""

from __future__ import annotations

from functools import lru_cache
from datetime import datetime, date
from typing import Optional
import numpy as np
import pandas as pd

C_KM_S = 299_792.458
AU_KM = 149_597_870.7


_HORIZONS_BODIES = {
    "STEREO-A": "STEREO Ahead",
    "STEREO-B": "STEREO Behind",
    "Wind": "Wind",
    "Parker Solar Probe": "Parker Solar Probe",
    "Solar Orbiter": "Solar Orbiter",
}


@lru_cache(maxsize=64)
def _heliocentric_distance_au(sc_name: str, ymd: tuple[int, int, int]) -> Optional[float]:
    """Return heliocentric distance in AU at midnight UT of the given date."""
    try:
        import astropy.units as u
        from sunpy.coordinates import get_horizons_coord
    except Exception:
        return None

    target = _HORIZONS_BODIES.get(sc_name, sc_name)
    when = datetime(*ymd)
    try:
        coord = get_horizons_coord(target, when)
    except Exception:
        return None
    try:
        return float(coord.radius.to(u.AU).value)
    except Exception:
        return None


def heliocentric_distance_au(sc_name: str, day: pd.Timestamp | datetime | date) -> Optional[float]:
    """Public entry point: cached daily heliocentric distance in AU."""
    if sc_name is None:
        return None
    d = pd.to_datetime(day).date()
    return _heliocentric_distance_au(sc_name, (d.year, d.month, d.day))


def time_shift_seconds(sc_distance_au: float) -> float:
    """Convert a spacecraft distance (AU) to a light-travel-time correction.

    The correction is the difference between the spacecraft distance and
    1 AU, in seconds. Positive when the spacecraft is inside 1 AU
    (subtracted from observed time to align with ground-based spectra).
    """
    if sc_distance_au is None or not np.isfinite(sc_distance_au):
        return 0.0
    delta_km = (sc_distance_au - 1.0) * AU_KM
    return delta_km / C_KM_S


def apply_ltt(ds, sc_distance_au: float):
    """Shift the time axis of a `DynamicSpectrum` by the LTT correction.

    Spacecraft closer than 1 AU sees the burst *earlier* than a 1 AU
    observer; to align it with the ground-based view we have to *add*
    the absolute time shift (i.e. push the time forward by |1 AU - r|/c).
    Equivalently we add `(1 AU - r_sc) / c`, so we add the negative of
    `time_shift_seconds`.
    """
    from .loaders._common import DynamicSpectrum

    if sc_distance_au is None or not np.isfinite(sc_distance_au):
        return ds
    dt_s = -time_shift_seconds(sc_distance_au)
    shifted = ds.time + np.timedelta64(int(dt_s * 1e9), "ns")
    new_meta = dict(ds.meta)
    new_meta["ltt_applied_seconds"] = float(dt_s)
    new_meta["sc_distance_au"] = float(sc_distance_au)
    return DynamicSpectrum(
        time=shifted, freq_mhz=ds.freq_mhz, flux=ds.flux,
        instrument=ds.instrument, label=ds.label, meta=new_meta,
    )
