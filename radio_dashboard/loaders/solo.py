"""Solar Orbiter / RPW loader (HFR + TNR) via sunpy.Fido or local cache.

The Solar Orbiter archive is reached through `sunpy.net.Fido`. If sunpy
is not installed or the search fails we fall back to whatever HFR/TNR
CDFs are already cached under `data_dir`.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum
from ..downloaders import download_solo_rpw


def load(day, data_dir, level: str = "l2", auto_download: bool = True) -> DynamicSpectrum:
    data_dir = Path(data_dir)

    if auto_download:
        try:
            download_solo_rpw(day, data_dir, level=level)
        except Exception:
            pass

    d = pd.to_datetime(day).date()
    files = sorted(data_dir.glob(f"solo_{level}_rpw-hfr-surv*{d:%Y%m%d}*.cdf"))
    files += sorted(data_dir.glob(f"solo_{level}_rpw-tnr-surv*{d:%Y%m%d}*.cdf"))
    if not files:
        raise FileNotFoundError(
            f"No Solar Orbiter RPW CDFs for {d} in {data_dir}. "
            "Install sunpy and enable auto-download, or place files manually."
        )

    pieces = []
    for path in files:
        t, f, p = _read_rpw(path)
        pieces.append((t, f, p))

    # Use the first receiver's time grid as reference (HFR is typically faster).
    t_ref = pieces[0][0]

    pieces_freq = []
    pieces_flux = []
    for (t, f, p) in pieces:
        df = pd.DataFrame(p, index=pd.to_datetime(t), columns=f)
        df = df.reindex(pd.to_datetime(t_ref), method="nearest")
        pieces_freq.append(f)
        pieces_flux.append(df.values)

    freq = np.concatenate(pieces_freq)
    flux = np.concatenate(pieces_flux, axis=1)
    order = np.argsort(freq)
    return DynamicSpectrum(
        time=t_ref, freq_mhz=freq[order], flux=flux[:, order],
        instrument="solo", label="Solar Orbiter / RPW",
        meta={"units": "V^2/Hz", "cmap_hint": "Spectral_r", "yscale_hint": "log"},
    )


def _read_rpw(path):
    """Return (time, freq_mhz, flux shape (n_time, n_freq))."""
    try:
        from spacepy import pycdf
        with pycdf.CDF(str(path)) as cdf:
            t = np.asarray(cdf["Epoch"][...])
            # RPW HFR/TNR conventions: 'FREQUENCY' in kHz, 'AGC1' / 'PSD' for flux.
            f_var = "FREQUENCY" if "FREQUENCY" in cdf else "FREQ"
            psd_var = next((v for v in ("PSD", "AGC1", "VOLTAGE") if v in cdf), None)
            if psd_var is None:
                raise KeyError("no PSD/AGC1/VOLTAGE variable in CDF")
            f = np.asarray(cdf[f_var][...], dtype=float)
            p = np.asarray(cdf[psd_var][...], dtype=float)
    except Exception:
        import cdflib
        cdf = cdflib.CDF(str(path))
        t = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))
        try:
            f = np.asarray(cdf.varget("FREQUENCY"), dtype=float)
        except Exception:
            f = np.asarray(cdf.varget("FREQ"), dtype=float)
        for v in ("PSD", "AGC1", "VOLTAGE"):
            try:
                p = np.asarray(cdf.varget(v), dtype=float)
                break
            except Exception:
                continue
        else:
            raise

    if f.ndim == 2:
        f = f[0]
    if np.nanmax(f) > 1e5:
        f = f / 1e3  # kHz
    # RPW PSD usually V^2/Hz; convert to MHz for the dashboard.
    return t, f / 1e3, p
