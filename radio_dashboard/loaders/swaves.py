"""STEREO-A / SWAVES Level-2 loader."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum
from ..downloaders import download_swaves


def load(day, data_dir, version: str = "v02",
         probe: str = "A", auto_download: bool = True,
         smooth_sigma: float = 1.0) -> DynamicSpectrum:
    """Load STEREO/SWAVES L2 combined CDF for one UTC day.

    probe = 'A' uses avg_intens_ahead; 'B' uses avg_intens_behind.
    Light gaussian smoothing along (freq, time) is applied to match the
    look used in the source notebook; set smooth_sigma=0 to disable.
    """
    data_dir = Path(data_dir)
    if auto_download:
        try:
            download_swaves(day, data_dir, version=version)
        except Exception:
            pass

    d = pd.to_datetime(day).date()
    candidates = sorted(data_dir.glob(f"stereo_level2_swaves_{d:%Y%m%d}*.cdf"))
    if not candidates:
        raise FileNotFoundError(
            f"No STEREO/SWAVES CDF for {d} under {data_dir}. "
            "Place a file manually or enable auto-download."
        )

    cdf_path = candidates[0]
    time, freq_khz, data = _read_cdf(cdf_path, probe=probe)

    # Per-channel mean subtraction along the time axis.
    bkg = np.nanmean(data, axis=0)
    data = data - bkg

    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            data = gaussian_filter(data, sigma=smooth_sigma)
        except Exception:
            pass

    return DynamicSpectrum(
        time=time,
        freq_mhz=freq_khz / 1e3,
        flux=data,
        instrument="swaves",
        label=f"STEREO-{probe} / SWAVES",
        meta={
            "units": "mean-subtracted intensity (a.u.)",
            "cmap_hint": "Spectral_r",
            "yscale_hint": "log",
            "file": str(cdf_path),
        },
    )


def _read_cdf(path: Path, probe: str = "A"):
    """Return (time, freq_khz, data shape (n_time, n_freq)).

    Tries spacepy.pycdf, falls back to cdflib if available.
    """
    var = "avg_intens_ahead" if probe.upper() == "A" else "avg_intens_behind"

    try:
        from spacepy import pycdf
        with pycdf.CDF(str(path)) as cdf:
            t = np.array(cdf.get("Epoch"))
            f = np.array(cdf.get("frequency"))
            d = np.array(cdf.get(var))
        return t, f, d.T if d.shape[0] != t.size else d
    except Exception:
        pass

    import cdflib
    cdf = cdflib.CDF(str(path))
    t = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))
    f = cdf.varget("frequency")
    d = cdf.varget(var)
    if d.shape[0] != len(t):
        d = d.T
    return np.asarray(t), np.asarray(f), np.asarray(d)
