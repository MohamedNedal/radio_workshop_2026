"""HTTP downloaders for the instruments with open archives.

Each downloader:

- accepts a date (or date range), a destination directory, and an
  `overwrite=False` flag.
- caches files under `dest / <fname>` and skips the request if the file
  already exists, unless `overwrite=True`.
- returns either a dict {receiver: Path} (multi-file) or a Path (single).

Network access is the only thing the loaders need from the outside world.
If `requests` is missing or the network is offline, callers fall back to
the local cache.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError:  # pragma: no cover - imported lazily in get_*
    requests = None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _ensure_requests():
    if requests is None:
        raise RuntimeError(
            "The 'requests' package is required for HTTP downloads but is "
            "not installed. `pip install requests`."
        )


def _as_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    fmt = "%Y-%m-%d" if "-" in str(d) else "%Y%m%d"
    return datetime.strptime(str(d), fmt).date()


def _stream_to(url: str, dest: Path, timeout: int = 60, chunk: int = 1 << 16) -> Path | None:
    _ensure_requests()
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for c in r.iter_content(chunk_size=chunk):
                    fh.write(c)
            tmp.replace(dest)
            return dest
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"download failed for {url}: {e}") from e


# ---------------------------------------------------------------------------
# STEREO / SWAVES
# ---------------------------------------------------------------------------

SWAVES_BASE = "https://spdf.gsfc.nasa.gov/pub/data/stereo/combined/swaves/level2_cdf"


def download_swaves(day, dest_dir, version: str = "v02",
                    overwrite: bool = False) -> Path | None:
    d = _as_date(day)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"stereo_level2_swaves_{d:%Y%m%d}_{version}.cdf"
    local = dest_dir / fname
    if local.exists() and not overwrite:
        return local
    url = f"{SWAVES_BASE}/{d:%Y}/{fname}"
    return _stream_to(url, local)


# ---------------------------------------------------------------------------
# Wind / WAVES
# ---------------------------------------------------------------------------

WIND_BASE = "https://spdf.gsfc.nasa.gov/pub/data/wind/waves"


def download_wind(day, dest_dir, level: str = "l2",
                  receivers: Iterable[str] = ("rad1", "rad2", "tnr"),
                  overwrite: bool = False) -> dict[str, Path | None]:
    """Return {receiver: Path or None}. Picks the highest-version match."""
    _ensure_requests()
    d = _as_date(day)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if level == "l2":
        out: dict[str, Path | None] = {}
        for instr in receivers:
            dir_url = f"{WIND_BASE}/{instr}_l2/{d:%Y}/"
            try:
                listing = requests.get(dir_url, timeout=30).text
            except Exception:
                out[instr] = None
                continue
            versions = re.findall(
                rf"wi_l2_wav_{instr}_{d:%Y%m%d}_v(\d+)\.cdf", listing
            )
            if not versions:
                out[instr] = None
                continue
            fname = f"wi_l2_wav_{instr}_{d:%Y%m%d}_v{max(versions, key=int)}.cdf"
            local = dest_dir / fname
            if local.exists() and not overwrite:
                out[instr] = local
            else:
                out[instr] = _stream_to(dir_url + fname, local)
        return out

    elif level == "h1":
        dir_url = f"{WIND_BASE}/wav_h1/{d:%Y}/"
        try:
            listing = requests.get(dir_url, timeout=30).text
        except Exception:
            return {"h1": None}
        versions = re.findall(rf"wi_h1_wav_{d:%Y%m%d}_v(\d+)\.cdf", listing)
        if not versions:
            return {"h1": None}
        fname = f"wi_h1_wav_{d:%Y%m%d}_v{max(versions, key=int)}.cdf"
        local = dest_dir / fname
        if local.exists() and not overwrite:
            return {"h1": local}
        return {"h1": _stream_to(dir_url + fname, local)}

    raise ValueError(f"unknown wind level: {level!r}")


# ---------------------------------------------------------------------------
# PSP / FIELDS RFS
# ---------------------------------------------------------------------------

PSP_BASE = "https://spdf.gsfc.nasa.gov/pub/data/psp/fields"


def download_psp(day, dest_dir, level: str = "l2", version: str = "v03",
                 receivers: Iterable[str] = ("lfr", "hfr"),
                 overwrite: bool = False) -> dict[str, Path | None]:
    if level not in ("l2", "l3"):
        raise ValueError(f"level must be 'l2' or 'l3', got {level!r}")
    d = _as_date(day)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path | None] = {}
    for rx in receivers:
        fname = f"psp_fld_{level}_rfs_{rx}_{d:%Y%m%d}_{version}.cdf"
        local = dest_dir / fname
        url = f"{PSP_BASE}/{level}/rfs_{rx}/{d:%Y}/{fname}"
        if local.exists() and not overwrite:
            out[rx] = local
            continue
        try:
            out[rx] = _stream_to(url, local)
        except RuntimeError:
            out[rx] = None
    return out


# ---------------------------------------------------------------------------
# NDA (real-time DAM mirror, daily 1-Hz FITS)
# ---------------------------------------------------------------------------

NDA_BASE = "https://realtime.obs-nancay.fr/dam/data_dam_affichage/data_dam"


def download_nda(day, dest_dir, overwrite: bool = False) -> Path | None:
    """Best-effort fetch of an NDA daily file. Filename convention can vary,
    so we scan the dated directory listing and grab the first FITS match.
    """
    _ensure_requests()
    d = _as_date(day)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # The mirror exposes a per-day index page; we look for any .fits link.
    dir_url = f"{NDA_BASE}/{d:%Y}/{d:%m}/{d:%d}/"
    try:
        listing = requests.get(dir_url, timeout=30).text
    except Exception:
        return None
    matches = re.findall(r'href="(NDA[^"]+\.fits)"', listing, flags=re.IGNORECASE)
    if not matches:
        return None
    fname = matches[0]
    local = dest_dir / fname
    if local.exists() and not overwrite:
        return local
    try:
        return _stream_to(dir_url + fname, local)
    except RuntimeError:
        return None


# ---------------------------------------------------------------------------
# Solar Orbiter / RPW - optional, via sunpy Fido if installed
# ---------------------------------------------------------------------------

def download_solo_rpw(day, dest_dir, level: str = "l2",
                      overwrite: bool = False) -> list[Path]:
    """Download Solar Orbiter / RPW HFR + TNR for one day via sunpy.Fido."""
    try:
        import astropy.units as u
        from sunpy.net import Fido, attrs as a
    except Exception as e:
        raise RuntimeError(
            "Solar Orbiter download requires sunpy; install with "
            "`pip install sunpy`."
        ) from e

    d = _as_date(day)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    tstart = f"{d.isoformat()}T00:00"
    tend   = f"{d.isoformat()}T23:59"

    query = Fido.search(
        a.Time(tstart, tend),
        a.Instrument("RPW"),
        a.Level(level.upper()),
    )
    files = Fido.fetch(query, path=str(dest_dir), overwrite=overwrite)
    return [Path(p) for p in files]
