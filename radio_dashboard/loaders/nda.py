"""Nancay Decameter Array loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum
from ..downloaders import download_nda


def load(day, data_dir, channel: str = "ch1", auto_download: bool = True,
         polarisation: str = "ch1") -> DynamicSpectrum:
    """NDA decametric spectrograph for one day.

    `channel`/`polarisation`:
        - 'ch1' : left circular
        - 'ch2' : right circular
        - 'diff': ch2 - ch1 (circular polarisation difference)
    """
    data_dir = Path(data_dir)
    if auto_download:
        try:
            download_nda(day, data_dir)
        except Exception:
            pass

    d = pd.to_datetime(day).date()
    files = sorted(data_dir.glob(f"*{d:%Y%m%d}*.fits")) or sorted(data_dir.glob("NDA_*.fits"))
    if not files:
        raise FileNotFoundError(
            f"No NDA FITS files in {data_dir}. Place a daily file manually."
        )

    from astropy.io import fits

    with fits.open(files[0]) as hdul:
        freq = np.asarray(hdul[3].data).reshape(-1).astype(float)
        ch1 = np.asarray(hdul[1].data).T  # (time, freq)
        ch2 = np.asarray(hdul[2].data).T
        start_str = hdul[0].header["DATE-OBS"] + " " + hdul[0].header["TIME-OBS"]
        end_str   = hdul[0].header["DATE-OBS"] + " " + hdul[0].header["TIME-END"]

    start = datetime.strptime(start_str, "%d/%m/%Y %H:%M:%S")
    end   = datetime.strptime(end_str,   "%d/%m/%Y %H:%M:%S")
    time = pd.date_range(start=start, end=end, periods=ch1.shape[0]).to_numpy()

    label_map = {"ch1": "NDA LH", "ch2": "NDA RH", "diff": "NDA RH - LH"}
    label = label_map.get(channel, "NDA")

    if channel == "diff":
        bkg1 = np.nanmedian(ch1, axis=0)
        bkg2 = np.nanmedian(ch2, axis=0)
        flux = (ch2 - bkg2) - (ch1 - bkg1)
    elif channel == "ch2":
        flux = ch2
    else:
        flux = ch1

    return DynamicSpectrum(
        time=time, freq_mhz=freq, flux=flux,
        instrument="nda", label=label,
        meta={"units": "intensity (a.u.)", "cmap_hint": "Spectral_r"},
    )
