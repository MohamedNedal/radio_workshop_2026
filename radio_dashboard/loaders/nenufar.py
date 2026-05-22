"""NenuFAR pre-exported pickle loader."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum


def load(day, data_dir, group: str = "typeII",
         product: str = "stokesI", resample: str = "1S") -> DynamicSpectrum:
    """NenuFAR Stokes I or V/I as a DynamicSpectrum.

    product:
        "stokesI"        - intensity, converted to dB (10*log10).
        "stokesV_over_I" - circular polarisation fraction, dimensionless.
    """
    data_dir = Path(data_dir)
    d = pd.to_datetime(day).date()
    date_compact = f"{d:%Y%m%d}"

    candidates = [p for p in sorted(data_dir.glob("*"))
                  if date_compact in p.name and p.name.endswith(f"{group}.pkl")]
    if not candidates:
        raise FileNotFoundError(f"No NenuFAR pickles for {d} {group} in {data_dir}")

    target_token = "stokesI" if product == "stokesI" else "stokesV_over_I"
    match = next((p for p in candidates if target_token in p.name), None)
    if match is None:
        raise FileNotFoundError(
            f"No NenuFAR file for {product} ({group}) on {d} in {data_dir}"
        )

    df = pd.read_pickle(match)
    if product == "stokesI":
        df = 10 * np.log10(df.where(df > 0))
        if resample:
            df = df.resample(resample, axis=1).mean() if df.columns.dtype.kind == "M" \
                 else df.resample(resample).mean()
        label = f"NenuFAR I ({group})"
        units = "dB (10 log10 I)"
    else:
        label = f"NenuFAR V/I ({group})"
        units = "V / I"

    # The NenuFAR pickles are (time, frequency); columns are frequency in MHz.
    return DynamicSpectrum(
        time=df.index.to_numpy(),
        freq_mhz=df.columns.to_numpy(dtype=float),
        flux=df.values,
        instrument="nenufar", label=label,
        meta={"units": units,
              "cmap_hint": "seismic" if product != "stokesI" else "Spectral_r",
              "yscale_hint": "linear",
              "vmin": -1 if product != "stokesI" else None,
              "vmax": 1 if product != "stokesI" else None},
    )
