"""OVRO-LWA / EOVSA calibrated FITS loader."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum


def load(day, data_dir) -> DynamicSpectrum:
    """Load the calibrated OVRO-LWA / EOVSA spectrum (Stokes I, SFU)."""
    from astropy.io import fits
    from astropy.time import Time

    data_dir = Path(data_dir)
    d = pd.to_datetime(day).date()
    files = sorted(data_dir.glob(f"*{d:%Y%m%d}*.fits")) or sorted(data_dir.glob("*.fits"))
    if not files:
        raise FileNotFoundError(f"No OVSA FITS file in {data_dir}")

    with fits.open(files[0]) as hdul:
        # Primary cube has Stokes axis 0; pick total intensity.
        spec_pol = np.asarray(hdul[0].data)[:, 0, :, :][0]
        freq_ghz = np.asarray(hdul["SFREQ"].data["SFREQ"])
        ut = hdul["UT"].data
        mjd = ut["mjd"]
        ms  = ut["time"]
        time_mjd = mjd + ms / 86_400_000.0
        time_dt = Time(time_mjd, format="mjd").to_datetime()

    return DynamicSpectrum(
        time=np.asarray(time_dt),
        freq_mhz=freq_ghz.astype(float) * 1e3,
        flux=spec_pol.T,
        instrument="ovsa", label="OVRO-LWA / EOVSA",
        meta={"units": "SFU", "cmap_hint": "Spectral_r",
              "yscale_hint": "log", "norm_hint": "log"},
    )
