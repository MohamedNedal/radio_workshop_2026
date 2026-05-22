"""Wind / WAVES loader (TNR + RAD1 + RAD2)."""

from __future__ import annotations

from pathlib import Path
import glob
import numpy as np
import pandas as pd

from ._common import DynamicSpectrum
from ..downloaders import download_wind


def load(day, data_dir, level: str = "l2", auto_download: bool = True) -> DynamicSpectrum:
    """Return a single combined Wind/WAVES dynamic spectrum.

    The three receivers are stitched onto a common time grid (TNR's grid
    by default, which is the slowest) and the frequency axis is sorted.
    For level='l2' the result is the raw PSD in V^2/Hz; for 'h1' it is
    the background-normalised E_VOLTAGE_* product.
    """
    data_dir = Path(data_dir)
    if auto_download:
        try:
            download_wind(day, data_dir, level=level)
        except Exception:
            pass

    d = pd.to_datetime(day).date()

    if level == "l2":
        files = {
            "tnr":  _newest(data_dir, f"wi_l2_wav_tnr_{d:%Y%m%d}_v*.cdf"),
            "rad1": _newest(data_dir, f"wi_l2_wav_rad1_{d:%Y%m%d}_v*.cdf"),
            "rad2": _newest(data_dir, f"wi_l2_wav_rad2_{d:%Y%m%d}_v*.cdf"),
        }
        if not all(files.values()):
            missing = [k for k, v in files.items() if v is None]
            raise FileNotFoundError(f"Missing Wind/WAVES L2 file(s) for {d}: {missing}")
        receivers = []
        for name in ("tnr", "rad1", "rad2"):
            t, f, psd = _read_l2(files[name], name)
            receivers.append((t, f, psd))
        return _stitch(receivers, level="l2", date=d)

    elif level == "h1":
        path = _newest(data_dir, f"wi_h1_wav_{d:%Y%m%d}_v*.cdf")
        if path is None:
            raise FileNotFoundError(f"Missing Wind/WAVES h1 file for {d}")
        return _read_h1(path, date=d)

    raise ValueError(f"unknown level {level!r}")


def _newest(directory: Path, pattern: str):
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _open_cdf(path):
    try:
        from spacepy import pycdf
        return ("pycdf", pycdf.CDF(str(path)))
    except Exception:
        import cdflib
        return ("cdflib", cdflib.CDF(str(path)))


def _var(cdf_pair, name):
    kind, cdf = cdf_pair
    if kind == "pycdf":
        return np.asarray(cdf[name][...])
    return np.asarray(cdf.varget(name))


def _close(cdf_pair):
    kind, cdf = cdf_pair
    if kind == "pycdf":
        try:
            cdf.close()
        except Exception:
            pass


def _read_l2(path, instrument):
    cdf = _open_cdf(path)
    t = _var(cdf, "Epoch")
    f = _var(cdf, "FREQUENCY").astype(float)
    if np.nanmax(f) > 1e5:  # given in Hz
        f = f / 1e3

    def _psd(var):
        kind, c = cdf
        if kind == "pycdf":
            arr = np.asarray(c[var][...], dtype=float)
            fill = c[var].attrs.get("FILLVAL", None)
        else:
            arr = np.asarray(c.varget(var), dtype=float)
            try:
                fill = c.varattsget(var).get("FILLVAL")
            except Exception:
                fill = None
        if fill is not None:
            arr[arr == fill] = np.nan
        arr[arr <= 0] = np.nan
        return arr

    if instrument == "tnr":
        psd = _psd("PSD_V2")
    else:
        psd = np.fmax.reduce([_psd("PSD_V2_S"), _psd("PSD_V2_SP"), _psd("PSD_V2_Z")])

    _close(cdf)
    # ensure ascending frequency
    order = np.argsort(f)
    return t, f[order], psd[:, order]


def _read_h1(path, date):
    cdf = _open_cdf(path)
    t = _var(cdf, "Epoch")
    f_t = _var(cdf, "Frequency_TNR")
    f_1 = _var(cdf, "Frequency_RAD1")
    f_2 = _var(cdf, "Frequency_RAD2")
    s_t = _var(cdf, "E_VOLTAGE_TNR").astype(float)
    s_1 = _var(cdf, "E_VOLTAGE_RAD1").astype(float)
    s_2 = _var(cdf, "E_VOLTAGE_RAD2").astype(float)
    _close(cdf)

    receivers = [(t, f_t, s_t), (t, f_1, s_1), (t, f_2, s_2)]
    return _stitch(receivers, level="h1", date=date)


def _stitch(receivers, level, date):
    """Concatenate three receivers onto the TNR time grid."""
    t_common = receivers[0][0]

    pieces_freq = []
    pieces_flux = []
    for (t, f, d) in receivers:
        df = pd.DataFrame(d, index=pd.to_datetime(t), columns=f)
        df = df.reindex(pd.to_datetime(t_common), method="nearest")
        pieces_freq.append(f)
        pieces_flux.append(df.values)

    freq = np.concatenate(pieces_freq)
    flux = np.concatenate(pieces_flux, axis=1)
    order = np.argsort(freq)
    freq = freq[order]
    flux = flux[:, order]

    return DynamicSpectrum(
        time=t_common,
        freq_mhz=freq / 1e3,
        flux=flux,
        instrument="wind",
        label="Wind / WAVES",
        meta={
            "units": ("V^2/Hz" if level == "l2" else "E_VOLTAGE (normalised)"),
            "cmap_hint": "jet" if level == "l2" else "Spectral_r",
            "yscale_hint": "log",
            "level": level,
        },
    )
