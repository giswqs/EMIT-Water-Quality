# EMIT Water-Quality Products (MoE-VAE)

Generate gridded water-quality products from NASA **EMIT** hyperspectral
imagery using a Mixture-of-Experts Variational Autoencoder (MoE-VAE). EMIT is
distributed as **L1B at-sensor radiance**, so each scene is first
atmospherically corrected with [**ACOLITE**](https://github.com/acolite/acolite)
to produce an L2W surface-reflectance (`Rrs`) product, then run through the
models. For each scene the workflow estimates three products and writes
validated Cloud Optimized GeoTIFFs (COGs), plus per-date GeoJSON catalogs that
mosaic each day's passes together:

| Product   | Variable    | Units      |
|-----------|-------------|------------|
| Chlorophyll-a | `chla`      | mg m⁻³ |
| Total Suspended Solids | `tss`       | g m⁻³  |
| CDOM absorption @ 440 nm | `acdom440`  | m⁻¹    |

## Pipeline

```
EMIT L1B radiance (.nc)  ──ACOLITE──►  L2W Rrs (.nc)  ──MoE-VAE──►  chla / tss / acdom
        download_*.py                run_acolite              COGs + JSON catalogs
```

## Project layout

```
EMIT-Water-Quality/
├── moe_vae/               # MoE-VAE model + EMIT inference/IO helpers
├── model/                 # Trained weights & scalers (Chl-a / TSS / aCDOM440)
├── download_data.py       # Download L1B scenes over a specified date range
├── download_latest.py     # Download the most recent L1B scene(s)
├── emit_processing.py     # Shared logic: ACOLITE / load_models / process_scene / COG
├── run_file.py            # Process a single scene (ACOLITE + inference)
├── run_folder.py          # Process every scene in a folder
├── make_json.py           # Build the per-date GeoJSON catalogs from the COGs
├── requirements.txt
└── README.md
```

By default, downloads, intermediate ACOLITE L2 output, and products are written
to the shared data drive:

```
/media/hdd/Data/EMIT/
├── data/      # input EMIT L1B_RAD / L1B_OBS granules
├── L2/        # intermediate ACOLITE L2R / L2W output (per scene)
├── output/    # generated products (COGs, in per-product subfolders)
└── json/      # per-date GeoJSON catalogs
```

Override these with `--output`, `--l2-dir`, and the `EMIT_DATA_DIR` environment
variable.

## Installation

Python 3.10+ with a CUDA-capable GPU recommended (CPU works but is slower).

```bash
pip install -r requirements.txt
```

### ACOLITE

The atmospheric-correction step uses **ACOLITE** (which ships its own bundled
Python and is *not* pip-installable). HyperCoast handles it for you:
`hypercoast.download_acolite` fetches a complete official release on first use
and `hypercoast.run_acolite` runs it, so **no manual install is required** —
the run scripts download ACOLITE automatically the first time.

To reuse an existing install (or control where it lives), set `ACOLITE_DIR`:

```bash
export ACOLITE_DIR=/path/to/acolite_py_linux
```

Pass `--acolite-dir` to point at a specific install, or `--no-download` to
fail instead of fetching ACOLITE when it is missing.

## NASA Earthdata credentials

Downloading requires a free [Earthdata](https://urs.earthdata.nasa.gov)
account. Store credentials in `~/.netrc`:

```
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
```

(or set the `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` environment variables).

## Usage

### 1. Download data

```bash
# Latest available scene over the region of interest
python download_latest.py

# The 5 most recent acquisition dates, up to 2 passes from each
python download_latest.py --dates 5 --passes-per-date 2

# Scenes over a specific date range (Gulf of Mexico by default)
python download_data.py 2025-04-01 2025-04-30
python download_data.py 2025-04-01 2025-04-30 --count 5
python download_data.py 2025-04-14 2025-04-14 --bbox -99 18 -78 42
```

Each `EMITL1BRAD` granule bundles the `L1B_RAD` radiance file and the matching
`L1B_OBS` geolocation file; both are downloaded and ACOLITE uses them together.

### 2. Process scenes

```bash
# A single scene (path, or a filename found in the data folder)
python run_file.py EMIT_L1B_RAD_001_20250414T200042_2510413_035.nc
python run_file.py /path/to/EMIT_L1B_RAD_...nc --output /path/to/output

# Every scene in a folder (defaults to /media/hdd/Data/EMIT/data -> .../output)
python run_folder.py
python run_folder.py /path/to/scenes --output /path/to/output

# Also write the per-date GeoJSON catalogs afterwards
python run_folder.py --json-dir /media/hdd/Data/EMIT/json
```

Both run ACOLITE first (writing intermediate L2W into `--l2-dir`) and then
inference. `run_folder.py` loads the models once and continues past individual
scene failures (reporting them in a summary). Scenes whose COGs already exist
are **skipped** unless `--overwrite` is passed, so an interrupted backfill can
simply be re-run — worth knowing, because ACOLITE is by far the slow step.
Use `--limit N` for a quick test run and `--write-nc` to also emit the merged
per-scene NetCDF.

### 3. Build the catalogs

```bash
python make_json.py /media/hdd/Data/EMIT/output --json-dir /media/hdd/Data/EMIT/json
```

`make_json.py` reads the COGs on disk, so it can be re-run at any time (after a
backfill, or once new scenes are added) without reprocessing anything.

## Outputs

Each scene keeps its own products. A COG is named after the input L1B granule
and placed in a per-product subfolder, so every pass of a date survives instead
of overwriting the others:

```
output/
├── chla/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
├── tss/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
└── acdom/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
```

### Per-date catalogs

For every acquisition date and product, `make_json.py` writes a
`json/<YYYYMMDD>_<product>.json` minimal GeoJSON `FeatureCollection`:

```json
{"type":"FeatureCollection","features":[{"bbox":[...],"assets":{"image":{"href":"..."}}}]}
```

Each feature is one EMIT pass, its `bbox` read from the COG and its `href`
pointing at the published copy on Hugging Face. EMIT images in narrow (~75 km)
swaths and crosses a region several times a day, so a date usually holds
several features — together they mosaic into that day's coverage.

### About the COGs

EMIT L2W swaths are rotated/curved in lon/lat, so the array's `(row, col)`
layout is not axis-aligned and cannot be written to a GeoTIFF directly.
Instead each product is gridded at its true `(lon, lat)` onto a regular
EPSG:4326 grid (~1 km, 0.01°) with `scipy.interpolate.griddata`. This
georeferences correctly and the interpolation fills the thin rotated-scan gaps
for a continuous coastal field, while leaving the open ocean / large cloud gaps
outside the data hull as nodata. Each GeoTIFF is written with internal tiling,
overviews and DEFLATE compression, then validated with `rio_cogeo`.

Inference is deterministic: the models run in `eval()` mode, which disables
the MoE noisy gating and makes the VAE use its latent mean, so re-running a
scene reproduces the same products.

### Bands

Each product uses a different spectral range (the closest EMIT band to each
target wavelength is selected at inference time):

- **Chl-a**: 403–723 nm (row-wise min-max normalization)
- **TSS**: 403–895 nm (robust scaler)
- **aCDOM440**: 403–701 nm (robust scaler)

## Automated daily products

A GitHub Actions workflow (`.github/workflows/daily.yml`) runs every day
(and on demand via *Run workflow*). It downloads the most recent EMIT scene,
runs ACOLITE + inference, and publishes the results to two places:

- the repository's **`EMIT-Data`** release (GeoTIFFs)
- the **Hugging Face dataset** — COGs under `data/<product>/`, catalogs under
  `json/`: https://huggingface.co/datasets/giswqs/EMIT-Water-Quality

Because each COG is named after its granule, products accumulate across dates
and passes; re-processing a granule replaces just that file.

The workflow targets a **self-hosted runner** because it processes multi-GB
EMIT L1B granules onto the local data drive. ACOLITE is downloaded
automatically by HyperCoast on first use (set the `ACOLITE_DIR` repository
variable to reuse an existing install). Set these secrets under **Settings →
Secrets and variables → Actions**:

- `EARTHDATA_USERNAME` — NASA Earthdata login
- `EARTHDATA_PASSWORD` — NASA Earthdata password
- `HF_TOKEN` — Hugging Face token with write access to the dataset

## Notes

- The `data/`, `L2/`, `output/` and `json/` folders (and the ACOLITE install)
  are git-ignored, so large scenes and products are never committed.
- The `model/` weights are required and bundled in the repository.
