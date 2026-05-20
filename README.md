# Radio_workshop_2026

A collection of Jupyter notebooks for plotting solar radio dynamic spectra
from ground- and space-based instruments. Each notebook is self-contained
and follows the same recipe: locate the data files for a chosen date, load
them into a `(time, frequency)` DataFrame, subtract a per-channel background,
and render the result with a unified Matplotlib style.

The notebooks were prepared for a multi-instrument multi-wavelength solar
radio workshop and are intended both as working tools and as worked
examples of how to read the file formats published by each observatory.

## Instruments covered

| Notebook | Instrument | Frequency coverage | File format |
|---|---|---|---|
| `plot_ilofar.ipynb` | I-LOFAR / REALTA | 10 - 90, 110 - 190, 210 - 270 MHz (modes 3/5/7) | `sigproc` filterbank (`.fil`) |
| `plot_lofar.ipynb`  | LOFAR core / international stations | LBA and HBA bands | FITS + JSON metadata |
| `plot_nda.ipynb`    | Nançay Decameter Array (NDA) | 10 - 80 MHz | FITS |
| `plot_nenufar.ipynb`| NenuFAR | 10 - 85 MHz | Pickle (Stokes I and V/I) |
| `plot_orfees.ipynb` | ORFEES (Nançay) | 144 - 1004 MHz, 5 sub-bands | FITS (`int_orf*.fts`) |
| `plot_ovsa.ipynb`   | OVRO-LWA / EOVSA | 20 - 88 MHz | FITS |
| `plot_psp.ipynb`    | Parker Solar Probe / FIELDS RFS | 10.5 kHz - 19.2 MHz (LFR + HFR) | CDF |
| `plot_swaves.ipynb` | STEREO-A / SWAVES | 2.6 kHz - 16.025 MHz | CDF |
| `plot_wind.ipynb`   | Wind / WAVES (TNR + RAD1 + RAD2) | 4 kHz - 13.825 MHz | CDF |

## Data sources

Most space-based instruments are mirrored at SPDF/CDAWeb; ground-based
instruments have their own archives.

- STEREO / SWAVES Level-2 (combined): `https://spdf.gsfc.nasa.gov/pub/data/stereo/combined/swaves/level2_cdf/`
- Wind / WAVES: `https://spdf.gsfc.nasa.gov/pub/data/wind/waves/`
- Parker Solar Probe / FIELDS RFS: `https://spdf.gsfc.nasa.gov/pub/data/psp/fields/`
- NDA: `https://realtime.obs-nancay.fr/dam/data_dam_affichage/data_dam/`
- ORFEES: `https://rsdb.obs-nancay.fr/data/orfees/`
- NenuFAR: data delivered by the NenuFAR pipeline (per-burst pickle exports)
- LOFAR (core/international): observatory archive or station-local exports
- I-LOFAR / REALTA: REALTA local archive (`.fil` filterbank)
- OVRO-LWA / EOVSA: OVRO/EOVSA team archives

The notebooks ship without bundled data. By default each notebook expects
files under `./sample_data/<instrument>/`. For STEREO / SWAVES and Wind /
WAVES the notebooks include a small helper that downloads the CDF on demand
if it is not already cached locally.

## Repository layout

```
Radio_workshop_2026/
|-- plot_ilofar.ipynb
|-- plot_lofar.ipynb
|-- plot_nda.ipynb
|-- plot_nenufar.ipynb
|-- plot_orfees.ipynb
|-- plot_ovsa.ipynb
|-- plot_psp.ipynb
|-- plot_swaves.ipynb
|-- plot_wind.ipynb
|-- sample_data/    # data files, organised per-instrument (created on first run)
|   |-- swaves/
|   |-- wind/
|   |-- psp/
|   |-- ...
`-- outputs/        # PNGs and intermediate products written by the notebooks
```

## Installation

The notebooks share a common Python stack. A reasonable environment can be
created with `conda` or `pip` from Python 3.10+:

```bash
# core scientific stack
pip install numpy pandas scipy matplotlib astropy

# CDF I/O (STEREO/SWAVES, Wind/WAVES, PSP/FIELDS)
pip install spacepy
# spacepy.pycdf needs the NASA CDF C library; see
# https://spacepy.github.io/install.html for platform notes.
# As an alternative, cdflib is pure Python:
# pip install cdflib

# I-LOFAR filterbank reader
pip install sigpyproc

# Wind/SWAVES/PSP loaders that prefer tplot
pip install pyspedas

# HTTP (only used by the SWAVES and Wind fetchers)
pip install requests
```

If you only intend to run a subset of the notebooks, you can skip the
loaders you do not need. The PSP and SWAVES notebooks include commented-out
`pyspedas` cells as an alternative to the manual `pycdf` route.

## Quick start

Open any notebook in JupyterLab and change the `mydate` variable at the top
to the date you want. The other configuration lives in the same cell:
`data_dir` for input files, `outputs` for plots.

For STEREO / SWAVES the `fetch_STAswaves(year, month, day)` helper checks
`data_dir` first and only hits the SPDF mirror if the file is not already
cached. For Wind / WAVES the `download_wind_waves` helper does the same for
all three receivers. The other notebooks expect the files to already be
present under `./sample_data/<instrument>/` because their archives are not
all open-access via simple HTTP.

A typical workflow looks like:

```python
mydate = '2024-05-14'
year, month, day = mydate.split('-')

data_dir = './sample_data/swaves'
outputs  = './outputs'
os.makedirs(outputs, exist_ok=True)

time_ste, freq_ste, smoothed_ste_A, ste_norm = fetch_STAswaves(year, month, day)
```

## Per-notebook notes

`plot_swaves.ipynb` - STEREO-A / SWAVES Level-2. Reads `Epoch`, `frequency`
(kHz) and `avg_intens_ahead` from the combined Level-2 CDF. The helper
applies a per-channel mean subtraction and a light Gaussian smoothing
(`sigma=1`) before returning. Set `avg_intens_behind` if you want
STEREO-B.

`plot_wind.ipynb` - Wind / WAVES TNR + RAD1 + RAD2. Two products are
supported via the `data_level` flag: `'l2'` (per-receiver LESIA L2 files,
raw PSD in V^2/Hz on a log colour scale, matching the CDAWeb look) and
`'h1'` (single combined Hi-Res file with background-normalised intensity).

`plot_psp.ipynb` - Parker Solar Probe / FIELDS RFS. Loads the LFR
(10.5 kHz - 1.7 MHz) and HFR (1.3 - 19.2 MHz) Level-2 CDFs separately,
converts the linear PSD to dB above a 10^-16 V^2/Hz floor following
Pulupa et al. 2020 ([doi:10.3847/1538-4365/ab5dc0](https://doi.org/10.3847/1538-4365/ab5dc0)),
and stitches the two bands onto a common time axis with the overlap region
deduplicated.

`plot_orfees.ipynb` - ORFEES Stokes I from the `int_orf*.fts` files. Stacks
the five sub-bands, resamples to 1 s, and subtracts the per-channel median.

`plot_nda.ipynb` - NDA decametric spectrograph. Two channels of left and
right circular polarisation, plotted individually and as their difference.

`plot_nenufar.ipynb` - NenuFAR pre-exported pickles. Stokes I converted to
dB, V/I plotted alongside. Filenames follow the convention
`<YYYYMMDD>_<group>_stokesI.pkl` and `<YYYYMMDD>_<group>_stokesV_over_I.pkl`.

`plot_lofar.ipynb` - LOFAR core/international stations. Reads the FITS
files produced by the standard pipeline (`HDU[0]` data, `HDU[1]` frequency
axis, `HDU[2]` time axis) with the accompanying JSON metadata. Both LBA
and HBA bands are supported via the `band` flag.

`plot_ilofar.ipynb` - I-LOFAR / REALTA filterbank (`.fil`). Combines LOFAR
modes 3, 5 and 7 into a single log-frequency axis with NaN gaps between
bands. Selectable Stokes parameter (`I` or `V`).

`plot_ovsa.ipynb` - OVRO-LWA / EOVSA calibrated solar spectrum (SFU). The
dynamic range spans several decades, so the plot uses a logarithmic colour
scale by default.

## Conventions

All notebooks share the same plotting style (300 dpi for saved figures,
white background, fixed font sizes), the same `subtract_background_median`
helper, and the same `(time, freq)` DataFrame convention with time as the
index. Outputs are written to `./outputs/<instrument>_dyspec_<date>.png`.

The matplotlib epoch is pinned to `1970-01-01T00:00:00` at the top of every
notebook to avoid the ~10 microsecond offsets seen in older versions when
plotting dense time series.

## Contributing and feedback

Issues and pull requests are welcome. If you add support for a new
instrument, please follow the existing notebook template so the workshop
material stays consistent.

## Acknowledgements

Workshop material prepared at the Dublin Institute for Advanced Studies
(DIAS). Please cite the relevant instrument teams when using their data
in publications.
