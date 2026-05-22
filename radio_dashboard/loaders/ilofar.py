"""I-LOFAR / REALTA filterbank loader."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum


def load(day, data_dir, stokes: str = "I", resample: str = "1S") -> DynamicSpectrum:
    """REALTA filterbank for one day, modes 3 + 5 + 7 stitched with NaN gaps."""
    from sigpyproc.readers import FilReader
    from astropy.time import Time
    import astropy.units as u

    data_dir = Path(data_dir)
    d = pd.to_datetime(day).date()
    date_compact = f"{d:%Y%m%d}"

    matches = [
        p for p in sorted(data_dir.glob("*.fil"))
        if date_compact in p.name.replace("-", "") and f"stokes{stokes}" in p.name
    ]
    if not matches:
        raise FileNotFoundError(f"No REALTA .fil for {d} (Stokes {stokes}) in {data_dir}")

    reader = FilReader(str(matches[0]))
    n_samples = reader.header.nsamples
    data = reader.read_block(start=0, nsamps=n_samples)

    tstart = Time(data.header.tstart, format="mjd")
    tarray = tstart + (np.arange(data.shape[1]) * data.header.tsamp * u.s)
    Tarray = pd.to_datetime([t.iso for t in tarray])

    freqs = data.header.chan_freqs
    new_freq = _freq_axis_with_gaps(freqs)

    safe = np.where(data > 0, data, np.nan)
    data_log = np.log10(safe)

    data2 = np.full((new_freq.shape[0], data_log.shape[1]), np.nan, dtype=float)
    data2[0:88]    = data_log[0:88]
    data2[145:345] = data_log[88:288]
    data2[404:]    = data_log[289:]

    # Use a fully sorted, monotonically increasing frequency axis for plotting.
    f_sorted_idx = np.argsort(new_freq)
    new_freq = new_freq[f_sorted_idx]
    data2 = data2[f_sorted_idx]

    df = pd.DataFrame(data2.T, index=Tarray, columns=new_freq)
    if resample:
        df = df.resample(resample).mean()

    return DynamicSpectrum(
        time=df.index.to_numpy(),
        freq_mhz=df.columns.to_numpy(dtype=float),
        flux=df.values,
        instrument="ilofar", label=f"I-LOFAR (Stokes {stokes})",
        meta={"units": "log10(power)", "cmap_hint": "Spectral_r", "yscale_hint": "log"},
    )


def _freq_axis_with_gaps(freqs):
    """Insert NaN-filled fillers between LOFAR modes 3 / 5 / 7."""
    gap1 = np.flipud(freqs[288] + (np.arange(59) * 0.390625))
    gap2 = np.flipud(freqs[88]  + (np.arange(57) * 0.390625))
    extra = 59 + 57 - 1
    new_freq = np.zeros(extra + freqs.shape[0])
    new_freq[0:88]    = freqs[0:88]
    new_freq[88:145]  = gap2[:57]
    new_freq[145:345] = freqs[88:288]
    new_freq[345:404] = gap1[:59]
    new_freq[404:]    = freqs[289:]
    return new_freq
