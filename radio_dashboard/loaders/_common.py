"""Shared dataclass and helpers used by every instrument loader.

A loader returns a `DynamicSpectrum` with three core arrays:

- `time`: 1-D numpy array of `numpy.datetime64[ns]`.
- `freq_mhz`: 1-D numpy array of frequencies in MHz, ascending.
- `flux`: 2-D numpy array of shape (n_time, n_freq).

Plus a small `meta` dict carrying anything instrument-specific that the
plotter or physics module might want (units, colour-scale hint, etc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class DynamicSpectrum:
    """Plain container for a dynamic spectrum."""

    time: np.ndarray
    freq_mhz: np.ndarray
    flux: np.ndarray
    instrument: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Cast time to datetime64[ns]; nothing in the rest of the tool depends
        # on the input dtype as long as it can be cast through pandas.
        import pandas as pd

        if not (isinstance(self.time, np.ndarray) and self.time.dtype.kind == "M"):
            self.time = pd.to_datetime(self.time).to_numpy()

        self.freq_mhz = np.asarray(self.freq_mhz, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)

        if self.flux.shape != (self.time.size, self.freq_mhz.size):
            # Try the transposed convention used in some of the source notebooks.
            if self.flux.shape == (self.freq_mhz.size, self.time.size):
                self.flux = self.flux.T
            else:
                raise ValueError(
                    f"flux shape {self.flux.shape} incompatible with "
                    f"(n_time={self.time.size}, n_freq={self.freq_mhz.size})"
                )

        # Enforce ascending frequency for predictable plotting.
        if self.freq_mhz.size > 1 and self.freq_mhz[0] > self.freq_mhz[-1]:
            self.freq_mhz = self.freq_mhz[::-1]
            self.flux = self.flux[:, ::-1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.flux.shape

    def time_range(self) -> tuple[np.datetime64, np.datetime64]:
        return self.time[0], self.time[-1]

    def crop_time(self, t_start, t_end) -> "DynamicSpectrum":
        import pandas as pd

        t0 = pd.to_datetime(t_start).to_datetime64()
        t1 = pd.to_datetime(t_end).to_datetime64()
        m = (self.time >= t0) & (self.time <= t1)
        return DynamicSpectrum(
            time=self.time[m],
            freq_mhz=self.freq_mhz,
            flux=self.flux[m, :],
            instrument=self.instrument,
            label=self.label,
            meta=dict(self.meta),
        )
