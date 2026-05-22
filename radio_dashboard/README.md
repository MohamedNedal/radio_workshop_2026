# radio_dashboard

An interactive Streamlit tool for plotting solar radio dynamic spectra
from multiple instruments on the same time axis, clicking burst features
to record (time, frequency) points, and deriving drift rate, radial
speed, and electron beam energy from those points under a chosen
coronal/interplanetary electron-density model.

The dashboard reuses the loader and background-subtraction logic from
the per-instrument notebooks in the parent repository
(`plot_swaves.ipynb`, `plot_wind.ipynb`, ...) and packages them behind a
single UI.

## Features

- Pick a start and end date/time for the observing window.
- Pick one or more instruments from a multi-select.
- Pick the data level (`l2`, `l3`, `h1`, ...) and version tag
  **independently for each selected instrument** — PSP can be on `l3 v03`
  while Wind is on `l2` and SWAVES is on `v02` in the same session.
- Download data from open archives (PSP/FIELDS, STEREO/SWAVES, Wind/WAVES,
  NDA, Solar Orbiter/RPW). Files are cached under `sample_data/<instrument>/`
  and re-used on subsequent runs.
- One Plotly subplot per instrument, sharing the X (time) axis, with
  zero vertical padding between panels and an in-panel legend chip
  identifying each instrument. The user can zoom, pan, box-zoom, and
  reset across the full figure.
- Click any subplot to record a (time, frequency) point in a table.
  Points are sorted, exportable to CSV, and can be undone individually.
- Six electron-density models for converting frequency to heliocentric
  distance: Newkirk x1/x2/x4, Saito 1970, Leblanc-Dulk-Bougeret 1998,
  Mann et al. 1999 (hydrostatic isothermal), Sittler & Guhathakurta 1999
  (coronal hole / fast wind).
- From clicked points the tool fits drift rate (MHz/s), radial speed
  (km/s), mean height (R_sun), beam kinetic energy (keV) and beta = v/c.
- Light-travel-time correction for space-based instruments: spacecraft
  heliocentric distance is looked up from JPL Horizons via
  `sunpy.coordinates.get_horizons_coord` and used to shift the time axis
  to align with ground-based receivers.
- Background subtraction: per-channel median (default), mean, minimum,
  user-defined quiet-time window, or none.
- Flux normalisation: per-channel z-score, min-max, log10, or "dB
  relative to per-channel median".
- Frequency and time cropping driven by the observing-window controls.
- Save and load session JSON (date range, instruments, choices, clicked
  points) so you can resume an analysis later.
- Export clicked points plus fitted quantities as a single CSV.

## Installation

The dashboard targets Python 3.10+.

```bash
cd radio_dashboard
pip install -r requirements.txt
```

Notes on optional dependencies:

- `cdflib` is a pure-Python CDF reader and is enough for STEREO/SWAVES,
  Wind/WAVES, PSP/FIELDS and Solar Orbiter/RPW. If you already have
  `spacepy` plus the NASA CDF C library, the loaders prefer it
  automatically (slightly faster).
- `sigpyproc` is only needed if you want to load I-LOFAR / REALTA
  filterbank files.
- `sunpy` is only used for the optional Solar Orbiter download via
  `Fido` and for the light-travel-time spacecraft lookup; the
  dashboard runs without it but with LTT correction effectively
  disabled.
- `streamlit-plotly-events` is what makes click-to-record work. The
  dashboard still renders without it but click capture is disabled.

## Running

From the repository root:

```bash
streamlit run radio_dashboard/app.py
```

The default data cache root is `<repo>/sample_data`; set
`RADIO_DASHBOARD_DATA_ROOT` to override.

## Workflow

1. Pick a UTC date in the sidebar, choose the instruments, and press
   **Load data**. Downloadable instruments (PSP, SWAVES, Wind, NDA,
   Solar Orbiter) will fetch missing files on the fly.
2. The figure renders with one panel per instrument, time on a shared
   X-axis. Zoom and pan as needed; the colour scale uses percentile
   clipping (defaults 5 - 99) controlled by the sidebar slider.
3. Click on a burst feature in any panel to record a (time, frequency)
   point. The point is appended to the table at the bottom of the page.
4. Pick a density model and a harmonic (1 = fundamental, 2 = harmonic
   emission). The dashboard inverts each frequency to a heliocentric
   distance and fits drift rate and radial speed.
5. Download clicked points plus the fitted quantities as a CSV, or save
   the full session as JSON to resume later.

## Instrument support

| Key       | Instrument                         | Bands                 | Auto-download |
|-----------|------------------------------------|-----------------------|---------------|
| `swaves`  | STEREO-A / SWAVES                  | 2.6 kHz - 16 MHz      | yes (SPDF)    |
| `wind`    | Wind / WAVES (TNR + RAD1 + RAD2)   | 4 kHz - 13.8 MHz      | yes (SPDF)    |
| `psp`     | Parker Solar Probe / FIELDS RFS    | 10.5 kHz - 19.2 MHz   | yes (SPDF)    |
| `solo`    | Solar Orbiter / RPW (HFR + TNR)    | 4 kHz - 16 MHz        | yes (Fido)    |
| `nda`     | Nançay Decameter Array             | 10 - 80 MHz           | yes (best effort) |
| `orfees`  | ORFEES (Nançay)                    | 144 - 1004 MHz        | no            |
| `lofar`   | LOFAR core / international         | LBA + HBA             | no            |
| `ilofar`  | I-LOFAR / REALTA                   | 10 - 270 MHz          | no            |
| `nenufar` | NenuFAR                            | 10 - 85 MHz           | no            |
| `ovsa`    | OVRO-LWA / EOVSA                   | 20 - 18000 MHz        | no            |

For the non-downloadable instruments, place files under
`sample_data/<key>/` following the convention of the corresponding
notebook.

## Density-model references

- Newkirk, G. 1961, *ApJ* 133, 983
- Saito, K. 1970, *Ann. Tokyo Astron. Obs.* 12, 53
- Leblanc, Y., Dulk, G. A., & Bougeret, J.-L. 1998, *Sol. Phys.* 183, 165
- Mann, G. et al. 1999, *A&A* 348, 614
- Sittler, E. C. & Guhathakurta, M. 1999, *ApJ* 523, 812

## Repository layout

```
radio_dashboard/
|-- app.py                Streamlit entry point
|-- config.py             Instrument registry and defaults
|-- downloaders.py        HTTP fetchers for the open archives
|-- density_models.py     Density profiles + plasma-frequency helpers
|-- physics.py            Drift rate, beam speed, kinetic energy fits
|-- ltt.py                Light-travel-time correction via Horizons
|-- plotting.py           Plotly multi-panel figure builder
|-- processing.py         Background subtraction, normalisation, cropping
|-- session.py            JSON session save/load, CSV exports
|-- requirements.txt
|-- loaders/
|   |-- __init__.py       Lazy loader registry
|   |-- _common.py        DynamicSpectrum dataclass
|   |-- swaves.py
|   |-- wind.py
|   |-- psp.py
|   |-- solo.py
|   |-- nda.py
|   |-- orfees.py
|   |-- lofar.py
|   |-- ilofar.py
|   |-- nenufar.py
|   `-- ovsa.py
`-- README.md
```

## Suggested extensions

Open ideas to add if the workshop wants more:

- Lasso / box selection of bands to fit drift rate over a region rather
  than a sparse click set.
- Direct overlay of model density curves on the figure as guide lines
  the user can drag.
- Type II band-splitting analysis (upper/lower band ratio -> shock
  Mach number under a density-jump assumption).
- Frequency calibration overlays for known persistent sources (Cas A,
  Cyg A) on LOFAR / I-LOFAR / NDA panels.
- Goesch / GOES X-ray + STIX lightcurves added as an extra panel.
