"""Plotly multi-panel dynamic-spectrum builder.

The figure has one subplot per instrument, sharing the X (time) axis so
panning and zooming in either direction is propagated. The vertical
padding between panels is removed; the legend chip inside each panel
identifies the instrument and substitutes for a per-panel title.

Click points are returned to the calling Streamlit app via
`streamlit_plotly_events` (see app.py). This module is only concerned
with building the figure - no Streamlit state lives here.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .loaders._common import DynamicSpectrum


def _format_ltt_offset(seconds: float) -> str:
    """Pretty-print an LTT offset as e.g. '+12m 30s' or '-2m 15s'."""
    if seconds is None or not np.isfinite(seconds) or seconds == 0:
        return ""
    sign = "+" if seconds >= 0 else "-"
    s = abs(seconds)
    mins, secs = divmod(s, 60)
    if mins >= 60:
        hrs, mins = divmod(mins, 60)
        return f"LTT {sign}{int(hrs)}h {int(mins)}m {secs:.0f}s"
    if mins >= 1:
        return f"LTT {sign}{int(mins)}m {secs:.0f}s"
    return f"LTT {sign}{secs:.1f}s"


def build_figure(
    spectra: Sequence[DynamicSpectrum],
    vmin_pct: float = 5.0,
    vmax_pct: float = 99.0,
    height_per_panel: int = 230,
    show_density_overlay: bool = False,
    overlay_curves: list[dict] | None = None,
) -> go.Figure:
    """Return a Plotly figure with one Heatmap per instrument.

    Each panel uses its own colour scale (clipped at vmin/vmax_pct
    percentiles of finite data) and its own y-axis range; only the
    x-axis is shared.

    overlay_curves: list of {"name": str, "t": array, "f_mhz": array,
    "row": int (1-based), "color": str} drawn on the matching subplot.
    Used for density-model frequency tracks at clicked points.
    """
    n = len(spectra)
    if n == 0:
        return _empty_message("Pick at least one instrument and click 'Load'.")

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        vertical_spacing=0.0,
        row_heights=[1.0 / n] * n,
    )

    for i, ds in enumerate(spectra, start=1):
        flux = ds.flux
        finite = flux[np.isfinite(flux)]
        if finite.size == 0:
            zmin, zmax = 0.0, 1.0
        else:
            zmin = float(np.nanpercentile(finite, vmin_pct))
            zmax = float(np.nanpercentile(finite, vmax_pct))
        # Override defaults from meta if the loader set them.
        if ds.meta.get("vmin") is not None:
            zmin = float(ds.meta["vmin"])
        if ds.meta.get("vmax") is not None:
            zmax = float(ds.meta["vmax"])

        cmap = ds.meta.get("cmap_hint", "Spectral_r")
        cmap = _plotly_cmap(cmap)
        # Convert times to ISO strings so plotly auto-detects a date axis
        # even if the input dtype is `object` (datetime objects coming out
        # of pycdf for example) or anything other than datetime64[ns].
        x = pd.to_datetime(ds.time).strftime("%Y-%m-%dT%H:%M:%S.%f").tolist()

        heatmap = go.Heatmap(
            x=x,
            y=ds.freq_mhz,
            z=ds.flux.T,
            colorscale=cmap,
            zmin=zmin, zmax=zmax,
            colorbar=dict(
                len=1.0 / n,
                y=1 - (i - 0.5) / n,
                yanchor="middle",
                thickness=10,
                tickfont=dict(size=9),
                title=dict(text=ds.meta.get("units", ""), font=dict(size=10), side="right"),
            ),
            name=ds.label,
            hovertemplate=(
                f"<b>{ds.label}</b><br>"
                "t = %{x|%Y-%m-%d %H:%M:%S}<br>"
                "f = %{y:.3f} MHz<br>"
                "z = %{z:.3g}<extra></extra>"
            ),
            showscale=True,
        )
        fig.add_trace(heatmap, row=i, col=1)

        # In-panel legend chip. Append the LTT offset for space-based
        # panels that have one applied.
        chip = f"<b>{ds.label}</b>"
        ltt_sec = ds.meta.get("ltt_applied_seconds")
        if ltt_sec is not None and ltt_sec != 0:
            chip += f"  <span style='color:#444'>({_format_ltt_offset(ltt_sec)})</span>"
        fig.add_annotation(
            x=0.01, xref=f"x{i if i > 1 else ''} domain",
            y=0.92, yref=f"y{i if i > 1 else ''} domain",
            text=chip,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.35)",
            borderwidth=1, borderpad=3,
            font=dict(size=11, color="black"),
            xanchor="left", yanchor="top",
        )

        # Force a date X-axis on this subplot. Plotly's auto-detection
        # can otherwise fall back to 'linear' when timestamps arrive as
        # objects or when the type is ambiguous across shared axes.
        fig.update_xaxes(type="date", row=i, col=1)

        # Frequency axis hints.
        yscale = ds.meta.get("yscale_hint", "log")
        fig.update_yaxes(
            type=yscale if yscale in ("log", "linear") else "log",
            row=i, col=1,
            title=dict(text="f (MHz)", font=dict(size=10)),
            showgrid=False,
            zeroline=False,
            ticks="outside",
            ticklen=4,
        )

    if overlay_curves:
        for c in overlay_curves:
            xs = pd.to_datetime(c["t"]).strftime("%Y-%m-%dT%H:%M:%S.%f").tolist()
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=c["f_mhz"],
                    mode="lines",
                    line=dict(color=c.get("color", "white"), width=2, dash="dash"),
                    name=c.get("name", "density overlay"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=c.get("row", 1), col=1,
            )

    fig.update_xaxes(
        title=dict(text="Time (UT)", font=dict(size=11)),
        row=n, col=1,
        type="date",
        showspikes=True, spikemode="across", spikecolor="grey", spikethickness=1,
    )

    fig.update_layout(
        height=max(360, height_per_panel * n),
        margin=dict(l=70, r=70, t=10, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        dragmode="zoom",
        hovermode="closest",
        uirevision="static",  # preserve zoom across reruns
    )
    return fig


def _empty_message(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=text, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="grey"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=320)
    return fig


def _plotly_cmap(name: str) -> str:
    """Translate matplotlib colormap names to the closest Plotly one."""
    mapping = {
        "Spectral_r": "Spectral_r",
        "jet": "Jet",
        "seismic": "RdBu_r",
    }
    return mapping.get(name, name)
