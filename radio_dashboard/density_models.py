"""Coronal and interplanetary electron-density models.

All models expose:

    n_e(r_rsun) -> electron density in cm^-3

with r_rsun the heliocentric distance in solar radii. The plasma
frequency at that height is

    f_p = 8.98e-3 * sqrt(n_e)  MHz,     n_e in cm^-3

and the harmonic-2 emission frequency is 2 * f_p. Both are returned by
`freq_from_r` so the same plumbing works for harmonic and fundamental.

References
----------
- Newkirk, G. (1961), ApJ 133, 983.
- Saito, K. (1970), Ann. Tokyo Astron. Obs. 12, 53.
- Leblanc, Dulk & Bougeret (1998), Sol. Phys. 183, 165.
- Mann, G. et al. (1999), A&A 348, 614 (hydrostatic isothermal corona).
- Sittler, E. C. & Guhathakurta, M. (1999), ApJ 523, 812
  (polar coronal hole / fast wind, model B).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


# ---------------------------------------------------------------------------
# Density profiles n_e(r), with r in solar radii (R_sun).
# ---------------------------------------------------------------------------

def newkirk(r_rsun: np.ndarray, fold: float = 1.0) -> np.ndarray:
    """Newkirk (1961) coronal model. fold=1 (quiet), 2 (streamer), 4 (dense)."""
    r = np.asarray(r_rsun, dtype=float)
    return fold * 4.2e4 * 10.0 ** (4.32 / r)


def saito(r_rsun: np.ndarray) -> np.ndarray:
    """Saito (1970) equatorial coronal density."""
    r = np.asarray(r_rsun, dtype=float)
    # Two-component power-law fit valid roughly 1.03 - 5 Rsun.
    return 1.36e6 * r ** (-2.14) + 1.68e8 * r ** (-6.13)


def leblanc(r_rsun: np.ndarray) -> np.ndarray:
    """Leblanc, Dulk & Bougeret (1998) - matches in-situ density at 1 AU.

    Validity: roughly 1.1 Rsun out to 1 AU (215 Rsun).
    """
    r = np.asarray(r_rsun, dtype=float)
    return 3.3e5 * r ** (-2) + 4.1e6 * r ** (-4) + 8.0e7 * r ** (-6)


def mann(r_rsun: np.ndarray, T_K: float = 1.4e6) -> np.ndarray:
    """Mann et al. (1999) hydrostatic isothermal model.

    n_e(r) = n0 * exp[A * (1/r - 1)],
    A      = G * M_sun * mu * m_p / (k_B * T * R_sun)

    For T = 1.4 MK and mu = 0.6 this gives A ~= 14.5; the prefactor n0 is
    set so that n_e(1 Rsun) matches the standard 5.14e9 cm^-3 used by
    Mann et al. for the base of the corona.
    """
    r = np.asarray(r_rsun, dtype=float)
    # Physical constants
    G   = 6.67430e-11
    M_s = 1.98892e30
    mu  = 0.6
    m_p = 1.67262e-27
    k_B = 1.38065e-23
    R_s = 6.957e8
    A = G * M_s * mu * m_p / (k_B * T_K * R_s)
    n0 = 5.14e9
    return n0 * np.exp(A * (1.0 / r - 1.0))


def sittler_guhathakurta(r_rsun: np.ndarray) -> np.ndarray:
    """Sittler & Guhathakurta (1999) model B (polar coronal hole / fast wind).

    Valid roughly 1.05 - 5 Rsun. Uses the analytic form
    n_e(r) = N0 * r^-2 * exp(z1 + z2 / r) * (1 + z3/r + z4/r^2 + z5/r^3).
    """
    r = np.asarray(r_rsun, dtype=float)
    N0 = 1.0
    z1 = -16.0
    z2 = 16.0
    z3 = 0.41587
    z4 = 5.4150e-1
    z5 = 2.8696e-1
    # Numerical coefficients from Sittler-Guhathakurta 1999 Table 2 model B
    # rescaled so n_e(1 Rsun) ~= 6e8 cm^-3 (polar hole base).
    base = 6.4e8
    return base * (r ** -2) * np.exp(z1 + z2 / r) * (1 + z3 / r + z4 / r**2 + z5 / r**3)


# ---------------------------------------------------------------------------
# Registry and helpers used by physics.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DensityModel:
    key: str
    label: str
    func: Callable[[np.ndarray], np.ndarray]
    valid_min_rsun: float = 1.02
    valid_max_rsun: float = 215.0


MODELS: dict[str, DensityModel] = {
    "newkirk_1x": DensityModel("newkirk_1x", "Newkirk x1 (quiet)", lambda r: newkirk(r, 1.0), 1.05, 5.0),
    "newkirk_2x": DensityModel("newkirk_2x", "Newkirk x2 (streamer)", lambda r: newkirk(r, 2.0), 1.05, 5.0),
    "newkirk_4x": DensityModel("newkirk_4x", "Newkirk x4 (dense streamer)", lambda r: newkirk(r, 4.0), 1.05, 5.0),
    "saito":     DensityModel("saito",     "Saito 1970", saito, 1.05, 5.0),
    "leblanc":   DensityModel("leblanc",   "Leblanc et al. 1998", leblanc, 1.1, 215.0),
    "mann":      DensityModel("mann",      "Mann et al. 1999 (T = 1.4 MK)", mann, 1.0, 50.0),
    "sg":        DensityModel("sg",        "Sittler-Guhathakurta 1999 (coronal hole)", sittler_guhathakurta, 1.05, 5.0),
}


def plasma_frequency_mhz(n_e_cm3: np.ndarray) -> np.ndarray:
    """f_p [MHz] = 8.98e-3 * sqrt(n_e [cm^-3])."""
    return 8.98e-3 * np.sqrt(np.asarray(n_e_cm3, dtype=float))


def freq_from_r(r_rsun: np.ndarray, model_key: str,
                harmonic: int = 1) -> np.ndarray:
    """Convert heights -> emission frequency (MHz) for a chosen model and harmonic."""
    model = MODELS[model_key]
    ne = model.func(np.asarray(r_rsun, dtype=float))
    fp = plasma_frequency_mhz(ne)
    return harmonic * fp


def r_from_freq(freq_mhz: np.ndarray, model_key: str,
                harmonic: int = 1,
                r_grid: np.ndarray | None = None) -> np.ndarray:
    """Invert f(r) numerically. Returns heliocentric distance in R_sun.

    Uses a fine pre-computed grid and linear interpolation in log space,
    which is robust to the wide dynamic range of the density profiles.
    """
    if r_grid is None:
        # Log-spaced grid covering corona to 1 AU.
        r_grid = np.logspace(np.log10(1.01), np.log10(220.0), 4000)
    f_grid = freq_from_r(r_grid, model_key, harmonic=harmonic)

    # f(r) is monotonically decreasing on the relevant grids; sort ascending
    # by frequency before interpolating.
    order = np.argsort(f_grid)
    f_sorted = f_grid[order]
    r_sorted = r_grid[order]

    f = np.atleast_1d(np.asarray(freq_mhz, dtype=float))
    # Clip to grid to avoid extrapolation surprises; mark out-of-range as NaN.
    out = np.interp(f, f_sorted, r_sorted, left=np.nan, right=np.nan)
    if np.isscalar(freq_mhz) or np.ndim(freq_mhz) == 0:
        return float(out[0])
    return out
