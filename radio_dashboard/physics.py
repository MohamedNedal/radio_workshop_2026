"""Physics derived from clicked (t, f) points.

The workflow assumed here is:

1. The user clicks N points on the dynamic spectrum, each with a time t_i
   and an emission frequency f_i in MHz.
2. They pick a density model and a harmonic (1 = fundamental, 2 = harmonic).
3. We invert f_i -> r_i with `density_models.r_from_freq`.
4. From (t_i, r_i) we derive:
     - drift rate df/dt in MHz/s (linear regression on f vs t),
     - radial speed dr/dt in km/s (linear regression on r vs t),
     - electron beam speed v_beam (= dr/dt) and inferred kinetic energy
       (relativistic) for type III interpretation,
     - type II shock speed (= dr/dt) by direct interpretation of the band.

The same routine handles both type II and type III - what differs is the
interpretation, not the mathematics. We surface drift rate, dr/dt, and
kinetic energy for every fit and let the user decide what is physical.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .density_models import r_from_freq


R_SUN_KM = 695_700.0


@dataclass
class BurstFit:
    """Result of fitting a straight line to a set of clicked points.

    Frequencies are in MHz, time-since-first-point in seconds, distance
    in solar radii (and km), speeds in km/s, energy in keV.
    """

    n_points: int
    t0: pd.Timestamp
    drift_mhz_per_s: float
    drift_std: float

    radial_speed_km_s: float | None
    radial_speed_std_km_s: float | None
    mean_height_rsun: float | None
    height_range_rsun: tuple[float, float] | None

    # Beam interpretation (type III): non-relativistic v from dr/dt.
    beam_kinetic_energy_keV: float | None
    beta: float | None  # v / c

    model_key: str
    harmonic: int


def _safe_seconds(times: pd.Series) -> np.ndarray:
    """Seconds since the first point in `times` as a float array."""
    t = pd.to_datetime(times).to_numpy()
    return (t - t[0]) / np.timedelta64(1, "s")


def fit_burst(points: pd.DataFrame,
              model_key: str = "newkirk_2x",
              harmonic: int = 1) -> BurstFit | None:
    """Fit drift rate and radial speed to clicked (t, f) points.

    `points` is a DataFrame with at least the columns `time` (datetime-like)
    and `freq_mhz` (float). Returns None if fewer than 2 valid points.
    """
    if points is None or len(points) < 2:
        return None

    df = points.dropna(subset=["time", "freq_mhz"]).copy()
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) < 2:
        return None

    t_s = _safe_seconds(df["time"])
    f   = df["freq_mhz"].to_numpy(dtype=float)

    # Linear fit in frequency-time (drift rate) with uncertainty.
    drift, drift_std = _linregress(t_s, f)

    # Convert each frequency to a heliocentric distance using the density model.
    r = r_from_freq(f, model_key=model_key, harmonic=harmonic)
    if np.all(np.isnan(r)):
        v_km_s = v_std = None
        mean_r = None
        h_range = None
        ek_keV = beta = None
    else:
        mask = ~np.isnan(r)
        if mask.sum() >= 2:
            slope_rsun_per_s, slope_std = _linregress(t_s[mask], r[mask])
            v_km_s = slope_rsun_per_s * R_SUN_KM
            v_std  = slope_std * R_SUN_KM
            mean_r = float(np.nanmean(r))
            h_range = (float(np.nanmin(r)), float(np.nanmax(r)))
            ek_keV, beta = kinetic_energy_from_speed(v_km_s)
        else:
            v_km_s = v_std = None
            mean_r = None
            h_range = None
            ek_keV = beta = None

    return BurstFit(
        n_points=len(df),
        t0=pd.to_datetime(df["time"].iloc[0]),
        drift_mhz_per_s=float(drift),
        drift_std=float(drift_std),
        radial_speed_km_s=v_km_s,
        radial_speed_std_km_s=v_std,
        mean_height_rsun=mean_r,
        height_range_rsun=h_range,
        beam_kinetic_energy_keV=ek_keV,
        beta=beta,
        model_key=model_key,
        harmonic=harmonic,
    )


def _linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope and its 1-sigma uncertainty. Intercept discarded."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        return float("nan"), float("nan")
    sxy = np.sum((x - x_mean) * (y - y_mean))
    slope = sxy / sxx
    if n == 2:
        return float(slope), float("nan")
    resid = y - (slope * (x - x_mean) + y_mean)
    s2 = np.sum(resid ** 2) / (n - 2)
    se = np.sqrt(s2 / sxx)
    return float(slope), float(se)


def kinetic_energy_from_speed(v_km_s: float) -> tuple[float, float]:
    """Return (kinetic energy in keV, beta = v/c) from a speed in km/s.

    Uses the relativistic expression E_k = (gamma - 1) m_e c^2. For type
    III beams this is the usual diagnostic.
    """
    c_km_s = 299_792.458
    m_e_keV = 510.998_950
    if v_km_s is None or not np.isfinite(v_km_s) or v_km_s <= 0:
        return float("nan"), float("nan")
    beta = abs(v_km_s) / c_km_s
    if beta >= 1.0:
        # Clipping super-luminal artefacts from sparse fits.
        beta = 0.99999
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    return float((gamma - 1.0) * m_e_keV), float(beta)


def drift_rate_only(points: pd.DataFrame) -> tuple[float, float]:
    """Pure drift rate in MHz/s and its 1-sigma uncertainty.

    Convenience entry point that does not require a density model.
    """
    if points is None or len(points) < 2:
        return float("nan"), float("nan")
    df = points.dropna(subset=["time", "freq_mhz"]).sort_values("time")
    t_s = _safe_seconds(df["time"])
    return _linregress(t_s, df["freq_mhz"].to_numpy())
