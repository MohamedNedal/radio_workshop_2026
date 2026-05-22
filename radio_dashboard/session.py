"""Save/load session state and CSV export of clicked points.

Only JSON-serialisable state goes here (dates, strings, lists of floats).
The loaded `DynamicSpectrum` objects are deliberately not serialised:
they can always be regenerated from the cached data files plus the
session metadata.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Any
import pandas as pd


SESSION_VERSION = 1


def serialise_session(state: dict[str, Any]) -> str:
    """Serialise a slice of Streamlit session_state to JSON."""
    keep = {
        "version": SESSION_VERSION,
        "date_start": state.get("date_start"),
        "date_end": state.get("date_end"),
        "instruments": state.get("instruments"),
        # Per-instrument data settings (replaces the global level/version).
        "instrument_levels": state.get("instrument_levels", {}),
        "instrument_versions": state.get("instrument_versions", {}),
        "background_method": state.get("background_method"),
        "normalisation": state.get("normalisation"),
        "ltt_enabled": state.get("ltt_enabled"),
        "ltt_distances_au": state.get("ltt_distances_au"),
        "density_model": state.get("density_model"),
        "harmonic": state.get("harmonic"),
        "clicked_points": state.get("clicked_points"),
    }
    return json.dumps(keep, indent=2, default=_jsonable)


def _jsonable(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def deserialise_session(blob: bytes | str) -> dict[str, Any]:
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    data = json.loads(blob)
    if data.get("version") != SESSION_VERSION:
        # Forward-compatible: accept anyway, the app will validate keys.
        pass
    return data


def clicked_points_to_csv(points: pd.DataFrame | list[dict]) -> bytes:
    """Convert clicked points to a UTF-8 CSV blob."""
    if isinstance(points, list):
        df = pd.DataFrame(points)
    else:
        df = points.copy()
    if df.empty:
        df = pd.DataFrame(columns=["time", "freq_mhz", "instrument", "label"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_table_with_fit(points: pd.DataFrame, fit) -> bytes:
    """Concatenate the clicked-points table and the fit results into one CSV."""
    if points is None or len(points) == 0:
        return b"no clicked points\n"
    buf = io.StringIO()
    points.to_csv(buf, index=False)
    buf.write("\n# Fit results\n")
    if fit is None:
        buf.write("no fit (need >= 2 points)\n")
    else:
        rows = [
            ("n_points", fit.n_points),
            ("t0", fit.t0),
            ("drift_mhz_per_s", fit.drift_mhz_per_s),
            ("drift_std_mhz_per_s", fit.drift_std),
            ("radial_speed_km_s", fit.radial_speed_km_s),
            ("radial_speed_std_km_s", fit.radial_speed_std_km_s),
            ("mean_height_rsun", fit.mean_height_rsun),
            ("height_range_rsun", fit.height_range_rsun),
            ("beam_kinetic_energy_keV", fit.beam_kinetic_energy_keV),
            ("beta", fit.beta),
            ("density_model", fit.model_key),
            ("harmonic", fit.harmonic),
        ]
        for k, v in rows:
            buf.write(f"{k},{v}\n")
    return buf.getvalue().encode("utf-8")
