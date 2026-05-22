"""LOFAR core / international stations FITS loader."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum


def load(day, data_dir, sas_id: str | None = None, band: str = "LBA") -> DynamicSpectrum:
    """Combine all FITS blocks for one LOFAR observation."""
    from astropy.io import fits

    data_dir = Path(data_dir)
    d = pd.to_datetime(day).date()

    # Layout: <data_dir>/<SASID>_<YYYYMMDD>_<band>/LOFAR_<YYYYMMDD>_*.fits
    if sas_id:
        block_dir = data_dir / f"{sas_id}_{d:%Y%m%d}_{band}"
    else:
        # Best-effort: pick the first directory that matches the date.
        matches = [p for p in data_dir.iterdir() if p.is_dir() and f"{d:%Y%m%d}_{band}" in p.name]
        if not matches:
            raise FileNotFoundError(
                f"No LOFAR block directory for {d} ({band}) in {data_dir}"
            )
        block_dir = matches[0]

    fits_files = sorted(block_dir.glob(f"LOFAR_{d:%Y%m%d}_*.fits"))
    fits_files = [f for f in fits_files if not f.name.endswith(f"{band}_OUTER.fits")]
    if not fits_files:
        raise FileNotFoundError(f"No FITS blocks in {block_dir}")

    blocks = []
    for f in fits_files:
        with fits.open(f) as hdul:
            spec  = np.asarray(hdul[0].data)              # (freq, time)
            freqs = np.asarray(hdul[1].data["FREQ"])      # MHz
            times = pd.to_datetime(hdul[2].data["TIME"])
        blocks.append(pd.DataFrame(spec.T, index=times, columns=freqs))

    df = pd.concat(blocks, axis=0).sort_index()
    df = df.loc[~df.index.duplicated(keep="first")]

    return DynamicSpectrum(
        time=df.index.to_numpy(),
        freq_mhz=df.columns.to_numpy(dtype=float),
        flux=df.values,
        instrument="lofar", label=f"LOFAR / {band}",
        meta={"units": "intensity (a.u.)", "cmap_hint": "Spectral_r"},
    )
