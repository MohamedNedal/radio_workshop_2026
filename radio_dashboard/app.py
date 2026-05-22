"""Streamlit dashboard for multi-instrument solar radio dynamic spectra.

Run with:

    streamlit run radio_dashboard/app.py

The sidebar controls everything; the main pane hosts the interactive
Plotly figure (zoom/pan/box-zoom) and the clicked-points table.
"""

from __future__ import annotations

import io
import os
import sys
import traceback
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow `python -m streamlit run radio_dashboard/app.py` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radio_dashboard import config
from radio_dashboard.loaders import load as load_instrument
from radio_dashboard.loaders._common import DynamicSpectrum
from radio_dashboard.processing import (
    subtract_background, normalise, crop_freq, decimate,
)
from radio_dashboard.plotting import build_figure
from radio_dashboard.density_models import MODELS, freq_from_r, r_from_freq
from radio_dashboard.physics import fit_burst
from radio_dashboard.ltt import heliocentric_distance_au, apply_ltt
from radio_dashboard.session import (
    serialise_session, deserialise_session,
    clicked_points_to_csv, export_table_with_fit,
)

"""Click capture uses native Streamlit selection events (Streamlit 1.31+).

We previously depended on `streamlit-plotly-events`, which silently fails
on Plotly 6.x heatmaps (drops the z array and the figure renders blank).
The built-in `st.plotly_chart(on_select="rerun")` works correctly with
current Plotly versions.
"""


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Solar radio dynamic-spectrum dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<h3 style='margin-top:0.0rem'>Solar radio dynamic spectra "
    "<span style='color:grey;font-size:0.8em;font-weight:normal'> "
    "multi-instrument, multi-wavelength</span></h3>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "date_start": date(2024, 5, 14),
    "time_start": time(0, 0),
    "date_end":   date(2024, 5, 14),
    "time_end":   time(23, 59),
    "instruments": ["swaves", "wind", "psp"],
    # Per-instrument level/version dicts. Populated lazily from the
    # INSTRUMENTS registry the first time each instrument is selected.
    "instrument_levels": {},
    "instrument_versions": {},
    "background_method": "median",
    "normalisation": "none",
    "ltt_enabled": True,
    "ltt_distances_au": {},
    "density_model": "newkirk_2x",
    "harmonic": 1,
    "clicked_points": [],
    "loaded_spectra": {},
    "data_root": config.DEFAULT_DATA_ROOT,
    "vmin_pct": 5.0,
    "vmax_pct": 99.0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Sidebar - observing window
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Observing window")
    c1, c2 = st.columns(2)
    st.session_state.date_start = c1.date_input(
        "Start date", value=st.session_state.date_start,
    )
    st.session_state.time_start = c2.time_input(
        "Start time", value=st.session_state.time_start,
    )
    c1, c2 = st.columns(2)
    st.session_state.date_end = c1.date_input(
        "End date", value=st.session_state.date_end,
    )
    st.session_state.time_end = c2.time_input(
        "End time", value=st.session_state.time_end,
    )

    t_start = datetime.combine(st.session_state.date_start, st.session_state.time_start)
    t_end   = datetime.combine(st.session_state.date_end,   st.session_state.time_end)

    if t_end <= t_start:
        st.warning("End time must be after start time.")

    st.subheader("Instruments")
    st.session_state.instruments = st.multiselect(
        "Pick one or more",
        options=[k for k in config.DISPLAY_ORDER if k in config.INSTRUMENTS],
        format_func=lambda k: config.INSTRUMENTS[k].label,
        default=st.session_state.instruments,
    )

    st.session_state.data_root = st.text_input(
        "Data cache root",
        value=st.session_state.data_root,
    )

    # ----- Per-instrument data level / version -----
    if st.session_state.instruments:
        st.subheader("Data level / version (per instrument)")
        for key in st.session_state.instruments:
            instr = config.INSTRUMENTS[key]
            with st.expander(instr.label, expanded=False):
                # --- Level / product ---
                level_opts = list(instr.levels)
                if not level_opts:
                    # Instrument exposes a single product (e.g. ORFEES Stokes I,
                    # OVSA calibrated); nothing for the user to pick.
                    st.caption("Level / product: single product, no choice.")
                    st.session_state.instrument_levels[key] = None
                elif len(level_opts) == 1:
                    # Single option: surface it as text so the user knows
                    # what's selected without a misleading dropdown.
                    st.caption(f"Level / product: **{level_opts[0]}** "
                               "(only option for this instrument)")
                    st.session_state.instrument_levels[key] = level_opts[0]
                else:
                    cur_level = st.session_state.instrument_levels.get(key, level_opts[0])
                    if cur_level not in level_opts:
                        cur_level = level_opts[0]
                    st.session_state.instrument_levels[key] = st.selectbox(
                        "Level / product",
                        options=level_opts,
                        index=level_opts.index(cur_level),
                        key=f"level_{key}",
                        help=instr.notes or None,
                    )

                # --- Version ---
                version_opts = list(instr.versions)
                if not version_opts:
                    # No user-selectable version: the downloader either
                    # picks the newest available on the server, or the
                    # version isn't part of the filename.
                    st.caption(
                        "Version: auto-detected from the archive."
                        if instr.downloadable else
                        "Version: not used by this loader."
                    )
                    st.session_state.instrument_versions[key] = None
                else:
                    cur_version = st.session_state.instrument_versions.get(
                        key, version_opts[0]
                    )
                    use_custom = st.checkbox(
                        "Custom version",
                        value=(cur_version is not None and cur_version not in version_opts),
                        key=f"custom_ver_{key}",
                    )
                    if use_custom:
                        st.session_state.instrument_versions[key] = st.text_input(
                            "Version tag",
                            value=cur_version or version_opts[0],
                            max_chars=16,
                            key=f"ver_text_{key}",
                        )
                    else:
                        if cur_version not in version_opts:
                            cur_version = version_opts[0]
                        st.session_state.instrument_versions[key] = st.selectbox(
                            "Version",
                            options=version_opts,
                            index=version_opts.index(cur_version),
                            key=f"ver_sel_{key}",
                        )

    st.subheader("Processing")
    st.session_state.background_method = st.selectbox(
        "Background subtraction",
        ["none", "median", "mean", "min", "window"],
        index=1,
    )
    if st.session_state.background_method == "window":
        wstart = st.time_input("BG window start", value=time(0, 0), key="bgw_start")
        wend   = st.time_input("BG window end",   value=time(0, 30), key="bgw_end")
        st.session_state.bg_window = (
            datetime.combine(st.session_state.date_start, wstart),
            datetime.combine(st.session_state.date_start, wend),
        )
    st.session_state.normalisation = st.selectbox(
        "Flux normalisation",
        ["none", "zscore", "minmax", "log10", "db_relative"],
        index=0,
    )

    cmin, cmax = st.slider(
        "Colour scale percentile (low, high)",
        min_value=0.0, max_value=100.0, value=(5.0, 99.0), step=0.5,
    )
    st.session_state.vmin_pct = cmin
    st.session_state.vmax_pct = cmax

    st.subheader("Light-travel-time correction")
    st.session_state.ltt_enabled = st.checkbox(
        "Apply LTT to space-based panels",
        value=st.session_state.ltt_enabled,
    )

    st.subheader("Density model")
    st.session_state.density_model = st.selectbox(
        "Coronal / IP density",
        options=list(MODELS.keys()),
        format_func=lambda k: MODELS[k].label,
        index=list(MODELS.keys()).index(st.session_state.density_model),
    )
    st.session_state.harmonic = st.radio(
        "Emission",
        options=[1, 2],
        horizontal=True,
        index=[1, 2].index(st.session_state.harmonic),
        format_func=lambda h: (
            "Fundamental (f = fp)" if h == 1 else "Harmonic (f = 2 fp)"
        ),
    )

    st.subheader("Session")
    sess_blob = serialise_session(dict(st.session_state))
    st.download_button(
        "Download session JSON",
        data=sess_blob, file_name="radio_dashboard_session.json",
        mime="application/json",
    )
    uploaded = st.file_uploader("Load session JSON", type=["json"])
    if uploaded is not None:
        try:
            blob = uploaded.read()
            restored = deserialise_session(blob)
            # Migrate legacy global level/version onto each instrument so
            # older session files keep working.
            legacy_level = restored.pop("data_level", None)
            legacy_version = restored.pop("data_version", None)
            for k, v in restored.items():
                if k == "version":
                    continue
                if k in ("date_start", "date_end") and isinstance(v, str):
                    v = pd.to_datetime(v).date()
                st.session_state[k] = v
            if legacy_level is not None or legacy_version is not None:
                for k in st.session_state.instruments:
                    if legacy_level is not None and legacy_level in config.INSTRUMENTS[k].levels:
                        st.session_state.instrument_levels.setdefault(k, legacy_level)
                    if legacy_version is not None:
                        st.session_state.instrument_versions.setdefault(k, legacy_version)
            st.success("Session restored. Click 'Load data' to fetch.")
        except Exception as e:
            st.error(f"Could not load session: {e}")

    load_clicked = st.button("Load data", type="primary")


# ---------------------------------------------------------------------------
# Loading and post-processing
# ---------------------------------------------------------------------------

def _level_for(key: str) -> dict:
    """Build the per-instrument loader kwargs from the per-instrument dicts.

    Each instrument has its own `level` and `version` choice in
    `st.session_state.instrument_levels` / `instrument_versions`. If a
    given instrument has not been touched in the sidebar yet, fall back
    to the first entry in its `INSTRUMENTS` registry tuple.
    """
    instr = config.INSTRUMENTS[key]
    default_level = (instr.levels[0] if instr.levels else None)
    default_version = (instr.versions[0] if instr.versions else None)

    lvl = st.session_state.instrument_levels.get(key, default_level)
    ver = st.session_state.instrument_versions.get(key, default_version)

    # Map the registry's "level / product" string onto each loader's
    # keyword argument. The translation is per-instrument because the
    # underlying concept differs (processing level for PSP, polarisation
    # for I-LOFAR, antenna band for LOFAR, product type for NenuFAR).
    if key == "psp":
        return {"level": lvl or "l3", "version": ver or "v03"}
    if key == "wind":
        return {"level": lvl if lvl in ("l2", "h1") else "l2"}
    if key == "swaves":
        return {"version": ver or "v02"}
    if key == "solo":
        return {"level": lvl or "l2"}
    if key == "lofar":
        return {"band": lvl if lvl in ("LBA", "HBA") else "LBA"}
    if key == "ilofar":
        return {"stokes": lvl if lvl in ("I", "V") else "I"}
    if key == "nenufar":
        return {"product": lvl if lvl in ("stokesI", "stokesV_over_I") else "stokesI"}
    if key == "nda":
        # NDA loader takes a channel arg; expose it via the level dropdown
        # when the user customised it.
        return {"channel": lvl} if lvl in ("ch1", "ch2", "diff") else {}
    return {}


def _post_process(ds: DynamicSpectrum) -> DynamicSpectrum:
    """Crop, background-subtract, normalise, then decimate for display."""
    # Stash the pre-crop time range so the diagnostics expander can show
    # what was actually in the file, regardless of what the user picked.
    if ds.time.size:
        ds.meta["raw_time_range"] = (
            pd.to_datetime(ds.time[0]).isoformat(),
            pd.to_datetime(ds.time[-1]).isoformat(),
        )
        ds.meta["raw_n_time"] = int(ds.time.size)
    ds = ds.crop_time(t_start, t_end)
    bg = st.session_state.background_method
    if bg != "none":
        kwargs = {"method": bg}
        if bg == "window":
            kwargs["window"] = st.session_state.get("bg_window")
        ds = subtract_background(ds, **kwargs)

    if st.session_state.normalisation != "none":
        ds = normalise(ds, method=st.session_state.normalisation)

    # LTT correction (only for space-based instruments).
    instr = config.INSTRUMENTS[ds.instrument]
    if st.session_state.ltt_enabled and instr.platform == "space" and instr.sc_name:
        distance = heliocentric_distance_au(instr.sc_name, t_start)
        if distance is not None:
            ds = apply_ltt(ds, distance)
            st.session_state.ltt_distances_au[ds.instrument] = distance

    return decimate(ds)


def _load_all():
    spectra: dict[str, DynamicSpectrum] = {}
    for key in st.session_state.instruments:
        kwargs = _level_for(key)
        data_dir = config.data_dir_for(key, st.session_state.data_root)
        try:
            ds = load_instrument(
                key, day=st.session_state.date_start,
                data_dir=data_dir, **kwargs,
            )
            spectra[key] = _post_process(ds)
        except Exception as e:
            st.error(f"[{config.INSTRUMENTS[key].label}] {e}")
            with st.expander(f"Traceback for {key}", expanded=False):
                st.code(traceback.format_exc())
    return spectra


if load_clicked:
    with st.spinner("Loading and processing..."):
        st.session_state.loaded_spectra = _load_all()
    if st.session_state.loaded_spectra:
        st.success(f"Loaded {len(st.session_state.loaded_spectra)} instrument(s).")
        # Summarise per-panel state. The empty-after-crop case is the most
        # common reason a 'loaded' figure looks blank, so it gets a
        # dedicated warning that names the raw file's time range.
        for k, ds in st.session_state.loaded_spectra.items():
            label = config.INSTRUMENTS[k].label
            if ds.flux.size == 0 or ds.time.size == 0:
                raw_range = ds.meta.get("raw_time_range")
                if raw_range:
                    st.warning(
                        f"{label}: 0 samples inside the observing window "
                        f"{t_start} - {t_end}. The data file actually covers "
                        f"{raw_range[0]} to {raw_range[1]} "
                        f"({ds.meta.get('raw_n_time', '?')} samples). "
                        "Adjust the date / time range, or replace the cached file."
                    )
                else:
                    st.warning(
                        f"{label}: the loader returned an empty spectrum. "
                        "Check the data file and the observing window."
                    )
                continue
            finite_frac = float(np.isfinite(ds.flux).mean())
            if finite_frac == 0.0:
                st.warning(
                    f"{label}: all values non-finite. Check the background "
                    "subtraction window or normalisation choice."
                )
            elif finite_frac < 0.05:
                st.info(
                    f"{label}: only {finite_frac:.1%} of cells finite "
                    f"(shape {ds.shape}). The panel may look sparse."
                )


# ---------------------------------------------------------------------------
# Main panel - plotly figure with click capture
# ---------------------------------------------------------------------------

spectra_in_order = [
    st.session_state.loaded_spectra[k]
    for k in config.DISPLAY_ORDER
    if k in st.session_state.loaded_spectra
]

# Build density-model overlay curves for any clicked points (one per row).
overlay = []
clicked_df = pd.DataFrame(st.session_state.clicked_points or [],
                          columns=["time", "freq_mhz", "instrument"])
if not clicked_df.empty and spectra_in_order:
    model = st.session_state.density_model
    harm = st.session_state.harmonic
    # Convert clicked frequencies to heights once.
    heights = r_from_freq(clicked_df["freq_mhz"].to_numpy(),
                          model_key=model, harmonic=harm)
    if np.isfinite(heights).any():
        # Build a smooth curve over the time span by linear extrapolation
        # of (t, r) and then mapping back to frequency on each panel.
        t_arr = pd.to_datetime(clicked_df["time"]).to_numpy()
        for row, ds in enumerate(spectra_in_order, start=1):
            f_curve = freq_from_r(heights, model_key=model, harmonic=harm)
            overlay.append({"name": MODELS[model].label, "t": t_arr,
                            "f_mhz": f_curve, "row": row, "color": "white"})

fig = build_figure(
    spectra_in_order,
    vmin_pct=st.session_state.vmin_pct,
    vmax_pct=st.session_state.vmax_pct,
    overlay_curves=overlay,
    time_range=(t_start, t_end),
)

event = st.plotly_chart(
    fig,
    use_container_width=True,
    key="radio_plot",
    on_select="rerun",
    selection_mode=("points",),
)

# Streamlit returns either a dict (with 'selection') or a SelectionState-like
# object depending on version; handle both shapes defensively.
selection = None
if event is not None:
    if isinstance(event, dict):
        selection = event.get("selection")
    else:
        selection = getattr(event, "selection", None)

new_pts = (selection or {}).get("points") if selection else None
if new_pts:
    # Identify which subplot was clicked from curve_number. Each instrument
    # contributes exactly one Heatmap trace, in the same order as
    # `spectra_in_order`; overlay scatter traces are appended afterwards.
    n_heatmaps = len(spectra_in_order)
    added = 0
    for pt in new_pts:
        x = pt.get("x")
        y = pt.get("y")
        if x is None or y is None:
            continue
        curve_idx = pt.get("curve_number")
        instr_key = ""
        if curve_idx is not None and 0 <= int(curve_idx) < n_heatmaps:
            instr_key = spectra_in_order[int(curve_idx)].instrument
        st.session_state.clicked_points.append({
            "time": pd.to_datetime(x).isoformat(),
            "freq_mhz": float(y),
            "instrument": instr_key,
        })
        added += 1
    if added:
        st.rerun()


# ---------------------------------------------------------------------------
# Clicked points table, density fit, exports
# ---------------------------------------------------------------------------

st.subheader("Clicked points")
col_l, col_r = st.columns([3, 2])

with col_l:
    if st.session_state.clicked_points:
        df_pts = pd.DataFrame(st.session_state.clicked_points)
        df_pts["time"] = pd.to_datetime(df_pts["time"])
        df_pts = df_pts.sort_values("time").reset_index(drop=True)
        st.dataframe(df_pts, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        if c1.button("Undo last point"):
            st.session_state.clicked_points = st.session_state.clicked_points[:-1]
            st.rerun()
        if c2.button("Clear all points"):
            st.session_state.clicked_points = []
            st.rerun()
        c3.download_button(
            "Download CSV", data=clicked_points_to_csv(df_pts),
            file_name="clicked_points.csv", mime="text/csv",
        )
    else:
        st.caption("Click on the figure to record (time, frequency) points.")

with col_r:
    if st.session_state.clicked_points and len(st.session_state.clicked_points) >= 2:
        df_pts = pd.DataFrame(st.session_state.clicked_points)
        df_pts["time"] = pd.to_datetime(df_pts["time"])
        fit = fit_burst(
            df_pts[["time", "freq_mhz"]],
            model_key=st.session_state.density_model,
            harmonic=st.session_state.harmonic,
        )
        if fit is not None:
            st.markdown("**Burst fit**")
            rows = [
                ("Drift rate", f"{fit.drift_mhz_per_s:+.4f} ± "
                              f"{fit.drift_std:.4f} MHz/s"),
                ("Mean height",
                 f"{fit.mean_height_rsun:.2f} Rsun"
                 if fit.mean_height_rsun is not None else "n/a"),
                ("Height range",
                 f"{fit.height_range_rsun[0]:.2f} - "
                 f"{fit.height_range_rsun[1]:.2f} Rsun"
                 if fit.height_range_rsun else "n/a"),
                ("Radial speed",
                 f"{fit.radial_speed_km_s:.0f} ± "
                 f"{fit.radial_speed_std_km_s:.0f} km/s"
                 if fit.radial_speed_km_s is not None else "n/a"),
                ("Beam kinetic energy",
                 f"{fit.beam_kinetic_energy_keV:.2f} keV "
                 f"(beta = {fit.beta:.3f})"
                 if fit.beam_kinetic_energy_keV is not None else "n/a"),
                ("Density model", MODELS[fit.model_key].label),
                ("Harmonic", f"f = {fit.harmonic} fp"),
            ]
            for k, v in rows:
                st.markdown(f"- **{k}**: {v}")

            st.download_button(
                "Download fit + points CSV",
                data=export_table_with_fit(df_pts, fit),
                file_name="burst_fit.csv", mime="text/csv",
            )
    else:
        st.caption("Add at least two points to fit a drift rate.")


# ---------------------------------------------------------------------------
# Footer - status hints
# ---------------------------------------------------------------------------

with st.expander("Diagnostics", expanded=False):
    rows = []
    for k in st.session_state.instruments:
        instr = config.INSTRUMENTS[k]
        ds = st.session_state.loaded_spectra.get(k)
        raw_range = (ds.meta.get("raw_time_range") if ds else None) or ("-", "-")
        n_raw = (ds.meta.get("raw_n_time") if ds else None) or 0
        post_n = ds.time.size if ds else 0
        rows.append({
            "key": k, "label": instr.label,
            "platform": instr.platform,
            "freq_range_MHz": f"{instr.freq_min_mhz}-{instr.freq_max_mhz}",
            "level": st.session_state.instrument_levels.get(k),
            "version": st.session_state.instrument_versions.get(k),
            "downloadable": instr.downloadable,
            "ltt_au": st.session_state.ltt_distances_au.get(k),
            "file_t_start": raw_range[0],
            "file_t_end": raw_range[1],
            "n_samples_raw": n_raw,
            "n_samples_post_crop": post_n,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Cache root: `{st.session_state.data_root}`")
