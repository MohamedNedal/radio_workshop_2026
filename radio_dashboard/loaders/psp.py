"""Parker Solar Probe / FIELDS RFS loader."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum
from ..downloaders import download_psp


PSD_FLOOR = 1e-16


def load(day, data_dir, level: str = "l3", version: str = "v03",
         auto_download: bool = True) -> DynamicSpectrum:
    """Combined LFR + HFR dynamic spectrum in dB above PSD_FLOOR."""
    data_dir = Path(data_dir)
    if auto_download:
        try:
            download_psp(day, data_dir, level=level, version=version)
        except Exception:
            pass

    d = pd.to_datetime(day).date()
    lfr = _newest(data_dir, f"psp_fld_{level}_rfs_lfr_{d:%Y%m%d}_*.cdf")
    hfr = _newest(data_dir, f"psp_fld_{level}_rfs_hfr_{d:%Y%m%d}_*.cdf")
    if lfr is None or hfr is None:
        raise FileNotFoundError(
            f"Missing PSP/RFS CDFs for {d} ({level}). lfr={lfr}, hfr={hfr}"
        )

    t_l, f_l, p_l = _read_rfs(lfr, level=level, receiver="lfr")
    t_h, f_h, p_h = _read_rfs(hfr, level=level, receiver="hfr")

    # LFR has the lower cadence; align HFR to it.
    df_l = pd.DataFrame(p_l, index=pd.to_datetime(t_l), columns=f_l)
    df_h = pd.DataFrame(p_h, index=pd.to_datetime(t_h), columns=f_h)
    df_h = df_h.reindex(df_l.index, method="nearest")

    df = pd.concat([df_l, df_h], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df.sort_index(axis=1)

    return DynamicSpectrum(
        time=df.index.to_numpy(),
        freq_mhz=df.columns.to_numpy(dtype=float),
        flux=df.values,
        instrument="psp",
        label="PSP / FIELDS RFS",
        meta={
            "units": "dB above 1e-16 V^2/Hz",
            "cmap_hint": "Spectral_r",
            "yscale_hint": "log",
            "level": level,
        },
    )


def _newest(directory, pattern):
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _read_rfs(path, level, receiver):
    psd_name = f"psp_fld_{level}_rfs_{receiver}_auto_averages_ch0_V1V2"

    try:
        from spacepy import pycdf
        with pycdf.CDF(str(path)) as cdf:
            psd_var = cdf[psd_name]
            epoch_name = psd_var.attrs["DEPEND_0"]
            freq_name  = psd_var.attrs["DEPEND_1"]
            t = np.asarray(cdf[epoch_name][...])
            f = np.asarray(cdf[freq_name][...], dtype=float)
            p = np.asarray(psd_var[...], dtype=float)
    except Exception:
        import cdflib
        cdf = cdflib.CDF(str(path))
        attrs = cdf.varattsget(psd_name)
        epoch_name = attrs["DEPEND_0"]
        freq_name  = attrs["DEPEND_1"]
        t = cdflib.cdfepoch.to_datetime(cdf.varget(epoch_name))
        f = np.asarray(cdf.varget(freq_name), dtype=float)
        p = np.asarray(cdf.varget(psd_name), dtype=float)

    if f.ndim == 2:
        f = f[min(1, f.shape[0] - 1)]
    freq_mhz = f / 1e6

    safe = np.where(p > 0, p, np.nan)
    db = 10.0 * np.log10(safe / PSD_FLOOR)
    if db.shape[0] != len(t) and db.shape[1] == len(t):
        db = db.T
    return t, freq_mhz, db
