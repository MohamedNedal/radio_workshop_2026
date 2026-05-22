"""Background subtraction, normalisation and cropping helpers.

All functions take and return `DynamicSpectrum` so they can be chained:

    ds = subtract_background(ds, method="median")
    ds = normalise(ds, method="zscore")
    ds = ds.crop_time(t0, t1)
"""

from __future__ import annotations

import numpy as np
from .loaders._common import DynamicSpectrum


def subtract_background(ds: DynamicSpectrum, method: str = "median",
                        window: tuple | None = None) -> DynamicSpectrum:
    """Remove a per-channel background from the dynamic spectrum.

    method
        "median" - per-channel median over the full time axis (default,
        matches the notebooks' `subtract_background_median`).
        "mean"   - per-channel mean.
        "min"    - per-channel minimum (useful for very clean spectra).
        "window" - per-channel median taken over `window=(t0, t1)` only,
        which is the textbook approach when a pre-burst quiet interval
        is available.
        "none"   - identity.
    """
    flux = ds.flux

    if method == "none":
        return ds

    if method == "window":
        if window is None:
            raise ValueError("method='window' needs a (t0, t1) window")
        import pandas as pd
        t0 = pd.to_datetime(window[0]).to_datetime64()
        t1 = pd.to_datetime(window[1]).to_datetime64()
        m = (ds.time >= t0) & (ds.time <= t1)
        if not m.any():
            raise ValueError("background window contains no samples")
        bkg = np.nanmedian(flux[m, :], axis=0)
    elif method == "mean":
        bkg = np.nanmean(flux, axis=0)
    elif method == "min":
        bkg = np.nanmin(flux, axis=0)
    else:  # "median"
        bkg = np.nanmedian(flux, axis=0)

    out = flux - bkg[np.newaxis, :]
    return DynamicSpectrum(
        time=ds.time, freq_mhz=ds.freq_mhz, flux=out,
        instrument=ds.instrument, label=ds.label, meta=dict(ds.meta),
    )


def normalise(ds: DynamicSpectrum, method: str = "none") -> DynamicSpectrum:
    """Renormalise the flux array.

    method
        "none"        - identity.
        "zscore"      - per-channel z-score: (x - mean) / std.
        "minmax"      - per-channel [0, 1] scaling.
        "log10"       - log10, replacing non-positive entries with NaN.
        "db_relative" - 10 * log10(x / per-channel median).
    """
    flux = ds.flux

    if method == "none":
        return ds

    if method == "zscore":
        mu = np.nanmean(flux, axis=0)
        sd = np.nanstd(flux, axis=0)
        sd = np.where(sd == 0, np.nan, sd)
        out = (flux - mu) / sd
    elif method == "minmax":
        lo = np.nanmin(flux, axis=0)
        hi = np.nanmax(flux, axis=0)
        denom = np.where(hi - lo == 0, np.nan, hi - lo)
        out = (flux - lo) / denom
    elif method == "log10":
        safe = np.where(flux > 0, flux, np.nan)
        out = np.log10(safe)
    elif method == "db_relative":
        ref = np.nanmedian(flux, axis=0)
        safe = np.where((flux > 0) & (ref > 0), flux / ref, np.nan)
        out = 10.0 * np.log10(safe)
    else:
        raise ValueError(f"unknown normalisation method: {method}")

    return DynamicSpectrum(
        time=ds.time, freq_mhz=ds.freq_mhz, flux=out,
        instrument=ds.instrument, label=ds.label, meta=dict(ds.meta),
    )


def crop_freq(ds: DynamicSpectrum, f_min_mhz: float, f_max_mhz: float) -> DynamicSpectrum:
    """Restrict the spectrum to [f_min, f_max] MHz."""
    m = (ds.freq_mhz >= f_min_mhz) & (ds.freq_mhz <= f_max_mhz)
    if not m.any():
        raise ValueError(f"no channels in [{f_min_mhz}, {f_max_mhz}] MHz")
    return DynamicSpectrum(
        time=ds.time, freq_mhz=ds.freq_mhz[m], flux=ds.flux[:, m],
        instrument=ds.instrument, label=ds.label, meta=dict(ds.meta),
    )


def decimate(ds: DynamicSpectrum, max_time: int = 2000,
             max_freq: int = 600) -> DynamicSpectrum:
    """Downsample a large dynamic spectrum for fast browser rendering.

    Pure striding (no averaging) - the rendered figure is for navigation
    and clicking; the underlying full-resolution arrays are kept on
    disk for any later quantitative use.
    """
    n_t, n_f = ds.flux.shape
    s_t = max(1, n_t // max_time)
    s_f = max(1, n_f // max_freq)
    if s_t == 1 and s_f == 1:
        return ds
    return DynamicSpectrum(
        time=ds.time[::s_t],
        freq_mhz=ds.freq_mhz[::s_f],
        flux=ds.flux[::s_t, ::s_f],
        instrument=ds.instrument, label=ds.label, meta=dict(ds.meta),
    )
