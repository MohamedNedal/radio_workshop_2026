"""ORFEES loader (Stokes I, B1 - B5 stacked)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum


def load(day, data_dir, resample: str = "1S") -> DynamicSpectrum:
    """ORFEES Stokes I dynamic spectrum for one day.

    `resample` is any pandas offset alias (e.g. '1S', '500ms').
    """
    from astropy.io import fits
    from astropy.time import Time
    import astropy.units as u

    data_dir = Path(data_dir)
    d = pd.to_datetime(day).date()
    files = sorted(data_dir.glob(f"int_orf{d:%Y%m%d}_*.fts"))
    if not files:
        files = sorted(data_dir.glob("int_orf*.fts"))
    if not files:
        raise FileNotFoundError(f"No ORFEES files in {data_dir}.")

    with fits.open(files[0]) as orfees:
        stokes_i = np.hstack([orfees[2].data[f"STOKESI_B{i}"] for i in range(1, 6)])
        obs_start = orfees[0].header["DATE-OBS"]
        times = Time(obs_start) + (orfees[2].data["TIME_B1"] / 1000.0) * u.s
        freqs = np.hstack([orfees[1].data[f"FREQ_B{i}"] for i in range(1, 6)])

    df = pd.DataFrame(
        stokes_i,
        index=[t.datetime for t in times],
        columns=freqs.astype(float).reshape(-1),
    )
    if resample:
        df = df.resample(resample).mean()

    return DynamicSpectrum(
        time=df.index.to_numpy(),
        freq_mhz=df.columns.to_numpy(dtype=float),
        flux=df.values,
        instrument="orfees", label="ORFEES",
        meta={"units": "SFU", "cmap_hint": "Spectral_r", "yscale_hint": "log"},
    )
