"""Shared EMIT water-quality processing logic.

This module holds the reusable pieces of the EMIT MoE-VAE inference workflow
so that the model weights only have to be loaded once and can be reused across
many scenes. Unlike PACE (which ships an analysis-ready L2 surface-reflectance
product), EMIT is distributed as L1B at-sensor radiance, so each scene is first
atmospherically corrected with **ACOLITE** to produce an L2W ``Rrs`` product
before inference:

* :func:`run_acolite` - L1B radiance -> L2W surface reflectance (``Rrs_*``).
* :func:`load_models` - build the chl-a / TSS / aCDOM models and scalers.
* :func:`infer_scene_maps` - run inference on one L2W scene (in memory).
* :func:`save_product_to_cog` - grid a swath product and write a valid COG.
* :func:`save_products_to_nc` - write a merged multi-variable NetCDF.
* :func:`process_scene` - end-to-end ACOLITE + inference + outputs.

Every scene keeps its own products: a COG is named after the input L1B granule
and placed in a per-product subfolder, so several passes on the same date all
survive::

    output/chla/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
    output/tss/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
    output/acdom/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif

Entry points:

* ``run_file.py`` processes a single L1B radiance scene.
* ``run_folder.py`` processes every L1B radiance scene in a folder.
* ``make_json.py`` builds the per-date GeoJSON catalogs from the COGs.
"""

import os
import re
import sys
import glob
import pickle
from pathlib import Path

import numpy as np
import torch
import hypercoast
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

# Resolve paths relative to this module so it can run from any location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "moe_vae"))

from MoE_VAE import *  # noqa: E402,F401,F403
from data_loading import *  # noqa: E402,F401,F403
from plot_and_save import *  # noqa: E402,F401,F403
from model_inference import (  # noqa: E402
    preprocess_and_infer_emit_minmax,
    preprocess_and_infer_emit_robust,
)

# ===========================================================================
# Band definitions (nm). EMIT has ~7.4 nm sampling, so the closest band to
# each target wavelength is selected at inference time. Each product uses a
# different spectral range, matching how the models were trained.
# ===========================================================================
BANDS_403_701 = [
    403,
    411,
    418,
    425,
    433,
    440,
    448,
    455,
    463,
    470,
    477,
    485,
    492,
    500,
    507,
    515,
    522,
    530,
    537,
    544,
    552,
    559,
    567,
    574,
    582,
    589,
    597,
    604,
    611,
    619,
    626,
    634,
    641,
    649,
    656,
    664,
    671,
    679,
    686,
    693,
    701,
]

BANDS_403_723 = BANDS_403_701 + [708, 716, 723]

BANDS_403_895 = BANDS_403_723 + [
    731,
    738,
    746,
    753,
    768,
    776,
    783,
    790,
    798,
    805,
    813,
    820,
    828,
    835,
    843,
    850,
    858,
    865,
    873,
    880,
    887,
    895,
]

# Per-product band selection (chl-a / TSS / aCDOM use different ranges).
PRODUCT_BANDS = {
    "chla": BANDS_403_723,
    "tss": BANDS_403_895,
    "acdom440": BANDS_403_701,
}

# Map dataset variable -> output subfolder label (aCDOM drops the "440").
PRODUCT_LABELS = {"chla": "chla", "tss": "tss", "acdom440": "acdom"}

# Base URL the published COGs are served from. The per-date JSON catalogs
# reference the COGs here, mirroring the local ``<product>/<granule>.tif``
# layout of the output folder.
HF_DATA_URL = (
    "https://huggingface.co/datasets/giswqs/EMIT-Water-Quality/resolve/main/data"
)

# ===========================================================================
# ACOLITE configuration. Atmospheric correction is run through HyperCoast's
# ``run_acolite`` (which locates the bundled ``dist/acolite/acolite``
# executable) and HyperCoast's ``download_acolite`` (which fetches a complete
# official release). Point ACOLITE_DIR at an existing install to skip the
# download; otherwise it is downloaded on first use.
# ===========================================================================
# `or` (not a get default) so an empty ACOLITE_DIR env var still falls back.
ACOLITE_DIR = os.environ.get("ACOLITE_DIR") or os.path.join(
    BASE_DIR, "acolite_py_linux"
)

# EMIT-specific ACOLITE processing settings (water/cirrus/TOA masks tuned for
# EMIT), appended to the per-scene ``inputfile``/``output`` lines.
EMIT_ACOLITE_SETTINGS = """\
polygon=None
limit=None
atmospheric_correction=True
l2w_parameters=Rrs_*
rgb_rhot=True
rgb_rhos=True
map_l2w=False
l2w_mask=True
l2w_mask_water_parameters=True
l2w_mask_wave=790
l2w_mask_threshold=0.08
l2w_mask_cirrus=True
l2w_mask_cirrus_wave=1014
l2w_mask_cirrus_threshold=0.12
l2w_mask_high_toa=True
l2w_mask_high_toa_threshold=10
l2w_mask_negative_rhow=True
l2w_mask_negative_wave_range=410,730
l2w_mask_smooth=True
"""


def _acolite_binary(acolite_dir):
    """Return the path to the ACOLITE executable inside an install dir.

    Args:
        acolite_dir (str): The extracted ACOLITE directory (e.g.
            ``.../acolite_py_linux``).

    Returns:
        str: Path to the ``dist/acolite/acolite`` (or ``acolite.exe``) binary.
    """
    exe = "acolite.exe" if acolite_dir.rstrip("/\\").endswith("win") else "acolite"
    return os.path.join(acolite_dir, "dist", "acolite", exe)


def ensure_acolite(acolite_dir=None, download=True):
    """Return a usable ACOLITE install dir, downloading one if necessary.

    Args:
        acolite_dir (str, optional): Candidate ACOLITE install directory.
            Defaults to ``ACOLITE_DIR``.
        download (bool): If the install is missing, fetch a complete official
            release with ``hypercoast.download_acolite`` (default True).

    Returns:
        str: A directory whose ``dist/acolite/acolite`` executable exists.

    Raises:
        FileNotFoundError: If the install is missing and ``download`` is False.
    """
    acolite_dir = acolite_dir or ACOLITE_DIR
    if os.path.exists(_acolite_binary(acolite_dir)):
        return acolite_dir
    if not download:
        raise FileNotFoundError(
            f"ACOLITE executable not found under {acolite_dir}. Set ACOLITE_DIR "
            "to an existing install, or allow download=True."
        )
    # download_acolite extracts to <outdir>/acolite_py_<os> and returns it.
    outdir = os.path.dirname(os.path.abspath(acolite_dir)) or "."
    print("ACOLITE not found; downloading a release ...")
    return hypercoast.download_acolite(outdir=outdir)


# Conda/GDAL environment variables that would otherwise leak into the ACOLITE
# subprocess. ACOLITE is a self-contained PyInstaller bundle with its own GDAL,
# PROJ and shared libraries; if it inherits the conda env's GDAL_DRIVER_PATH /
# LD_LIBRARY_PATH it loads conda's libgdal (which needs a newer libtiff than
# ACOLITE bundles) and fails. These are stripped for the subprocess only.
_ACOLITE_STRIP_ENV = (
    "GDAL_DRIVER_PATH",
    "GDAL_DATA",
    "GDAL_PLUGIN_PATH",
    "PROJ_LIB",
    "PROJ_DATA",
    "PROJ_NETWORK",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)


class _clean_acolite_env:
    """Temporarily strip conda/GDAL env vars so ACOLITE uses its own bundle.

    The vars are removed from ``os.environ`` on enter (so a subprocess spawned
    inside the block inherits a clean environment) and restored on exit.
    Already-loaded libraries in the current process are unaffected.
    """

    def __enter__(self):
        self._saved = {
            k: os.environ.pop(k) for k in _ACOLITE_STRIP_ENV if k in os.environ
        }
        return self

    def __exit__(self, *exc):
        os.environ.update(self._saved)
        return False


def run_acolite(input_nc, out_root, acolite_dir=None, download=True, reuse=True):
    """Atmospherically correct one EMIT L1B radiance scene with ACOLITE.

    ACOLITE reads the L1B ``RAD`` granule (and the matching ``OBS``
    geolocation granule in the same folder) and writes L2R / L2W products,
    including the ``Rrs_*`` surface-reflectance bands the models consume. The
    run is driven by an EMIT-tuned settings file and executed through
    ``hypercoast.run_acolite``.

    Args:
        input_nc (str): Path to the EMIT ``L1B_RAD`` NetCDF file.
        out_root (str): Root directory for ACOLITE output (a per-scene
            subfolder is created inside it).
        acolite_dir (str, optional): ACOLITE install directory. Defaults to
            ``ACOLITE_DIR``; downloaded automatically if missing.
        download (bool): Download ACOLITE if it is not already installed
            (default True).
        reuse (bool): If an L2W product already exists for this scene, return
            it instead of re-running ACOLITE (default True). Atmospheric
            correction is by far the slowest step and its output depends only
            on the input granule and settings, so re-running inference (for
            example at a different output resolution) does not need to repeat
            it. Pass False to force a fresh correction.

    Returns:
        dict: ``{"scene", "input_nc", "output_dir", "l2r_files", "l2w_files"}``.

    Raises:
        FileNotFoundError: If ACOLITE produces no L2W file.
    """
    scene = Path(input_nc).stem
    out_path = os.path.join(out_root, scene)

    if reuse:
        existing = sorted(glob.glob(os.path.join(out_path, "*L2W*.nc")))
        if existing:
            print(f"Reusing existing ACOLITE L2W for {scene}: {existing[0]}")
            return {
                "scene": scene,
                "input_nc": str(input_nc),
                "output_dir": out_path,
                "l2r_files": sorted(glob.glob(os.path.join(out_path, "*L2R*.nc"))),
                "l2w_files": existing,
            }

    acolite_dir = ensure_acolite(acolite_dir, download=download)
    os.makedirs(out_path, exist_ok=True)

    settings_file = os.path.join(out_path, f"{scene}_acolite_settings.txt")
    with open(settings_file, "w") as f:
        f.write(f"inputfile={input_nc}\noutput={out_path}\n{EMIT_ACOLITE_SETTINGS}")

    print("=" * 80)
    print(f"Running ACOLITE for EMIT RAD: {input_nc}")
    print("=" * 80)
    # out_dir is passed explicitly: hypercoast.run_acolite requires it even
    # when a settings file is supplied. The clean-env block keeps conda's GDAL
    # out of ACOLITE's bundled runtime.
    with _clean_acolite_env():
        hypercoast.run_acolite(
            acolite_dir, settings_file=settings_file, out_dir=out_path
        )

    l2w_files = sorted(glob.glob(os.path.join(out_path, "*L2W*.nc")))
    l2r_files = sorted(glob.glob(os.path.join(out_path, "*L2R*.nc")))
    if not l2w_files:
        raise FileNotFoundError(f"No L2W file generated in: {out_path}")

    print(f"ACOLITE finished. L2W: {l2w_files[0]}")
    return {
        "scene": scene,
        "input_nc": str(input_nc),
        "output_dir": out_path,
        "l2r_files": l2r_files,
        "l2w_files": l2w_files,
    }


def _build_model(
    input_dim, encoder_hidden_dims, decoder_hidden_dims, use_softplus_output, device
):
    """Construct a MoE-VAE with the EMIT product architecture.

    Args:
        input_dim (int): Number of input spectral bands.
        encoder_hidden_dims (list[int]): Encoder hidden layer widths.
        decoder_hidden_dims (list[int]): Decoder hidden layer widths.
        use_softplus_output (bool): Whether to apply a softplus on the output
            (used for the chl-a model).
        device (torch.device): Device to place the model on.

    Returns:
        MoE_VAE: The constructed (untrained) model on ``device``.
    """
    return MoE_VAE(  # noqa: F405
        input_dim=input_dim,
        output_dim=1,
        latent_dim=16,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        activation="leakyrelu",
        use_norm="layer",
        use_dropout=False,
        use_softplus_output=use_softplus_output,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ).to(device)


def load_models(model_dir, device):
    """Build the chl-a, TSS and aCDOM models and load their weights/scalers.

    Args:
        model_dir (str): Directory containing the ``Chl-a``, ``TSS`` and
            ``aCDOM440`` model subfolders.
        device (torch.device): Device to load the models onto.

    Returns:
        dict: Mapping of product name to a dict with the loaded ``model`` and,
            for TSS/aCDOM, the ``scaler_Rrs`` and ``scaler_dict`` objects.
    """
    chla_model = _build_model(
        input_dim=len(PRODUCT_BANDS["chla"]),
        encoder_hidden_dims=[128, 64, 32],
        decoder_hidden_dims=[32, 64, 128],
        use_softplus_output=True,
        device=device,
    )
    chla_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "Chl-a", "best_model_minloss.pth"),
            map_location=device,
        )
    )

    tss_model = _build_model(
        input_dim=len(PRODUCT_BANDS["tss"]),
        encoder_hidden_dims=[128, 64, 32],
        decoder_hidden_dims=[32, 64, 128],
        use_softplus_output=False,
        device=device,
    )
    tss_dir = os.path.join(model_dir, "TSS")
    tss_model.load_state_dict(
        torch.load(os.path.join(tss_dir, "best_model_minloss.pth"), map_location=device)
    )
    with open(os.path.join(tss_dir, "scalers_Rrs_real.pkl"), "rb") as f:
        tss_scaler_Rrs = pickle.load(f)
    tss_scaler_dict = torch.load(
        os.path.join(tss_dir, "scaler.pt"), map_location="cpu", weights_only=False
    )

    acdom_model = _build_model(
        input_dim=len(PRODUCT_BANDS["acdom440"]),
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[32, 64],
        use_softplus_output=False,
        device=device,
    )
    acdom_dir = os.path.join(model_dir, "aCDOM440")
    acdom_model.load_state_dict(
        torch.load(
            os.path.join(acdom_dir, "best_model_minloss.pth"), map_location=device
        )
    )
    with open(os.path.join(acdom_dir, "scalers_Rrs_real.pkl"), "rb") as f:
        acdom_scaler_Rrs = pickle.load(f)
    acdom_scaler_dict = torch.load(
        os.path.join(acdom_dir, "scaler.pt"), map_location="cpu", weights_only=False
    )

    # eval() mode: disables noisy gating and makes the VAE use the latent
    # mean (deterministic inference).
    for mdl in (chla_model, tss_model, acdom_model):
        mdl.eval()

    return {
        "chla": {"model": chla_model},
        "tss": {
            "model": tss_model,
            "scaler_Rrs": tss_scaler_Rrs,
            "scaler_dict": tss_scaler_dict,
        },
        "acdom440": {
            "model": acdom_model,
            "scaler_Rrs": acdom_scaler_Rrs,
            "scaler_dict": acdom_scaler_dict,
        },
    }


def save_product_to_cog(
    out_tif,
    lat_2d,
    lon_2d,
    values_2d,
    resolution_m=60,
    method="linear",
    nodata=-9999.0,
):
    """Grid an EMIT swath product onto a regular grid and write a COG.

    EMIT L2W swaths are rotated/curved in lon/lat, so the array's (row, col)
    layout is not axis-aligned and cannot be written to a GeoTIFF directly.
    Each pixel is gridded at its true (lon, lat) onto a regular EPSG:4326 grid
    with ``scipy.interpolate.griddata``. This georeferences correctly and the
    linear interpolation fills the thin rotated-scan gaps for a continuous
    coastal field, while leaving open ocean / large cloud gaps as nodata
    (outside the data hull). The result is written as a Cloud Optimized
    GeoTIFF (internal tiling, overviews, DEFLATE compression) and validated.

    Args:
        out_tif (str): Output GeoTIFF path.
        lat_2d (np.ndarray): Latitude (degrees north, EPSG:4326).
        lon_2d (np.ndarray): Longitude (degrees east, EPSG:4326).
        values_2d (np.ndarray): Product values aligned with lat/lon (NaN for
            invalid pixels).
        resolution_m (float): Target grid resolution in metres. Defaults to 60,
            EMIT's native ground sampling distance, so gridding re-projects the
            swath without throwing away spatial detail. (PACE uses 1000 m
            because that is *its* native resolution; reusing that here would
            collapse ~280 EMIT pixels into every output cell.)
        method (str): ``griddata`` interpolation method (default "linear").
        nodata (float): Value used for empty cells.

    Returns:
        str: The path to the validated COG.
    """
    from scipy.interpolate import griddata

    lat = np.asarray(lat_2d, dtype=np.float64).ravel()
    lon = np.asarray(lon_2d, dtype=np.float64).ravel()
    val = np.asarray(values_2d, dtype=np.float64).ravel()

    geo_ok = np.isfinite(lat) & np.isfinite(lon)
    if not (geo_ok & np.isfinite(val)).any():
        raise ValueError(f"No valid pixels to grid for {out_tif}")
    lat, lon, val = lat[geo_ok], lon[geo_ok], val[geo_ok]

    # Regular grid spanning the swath extent; metres -> degrees at scene centre.
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_c = (lat_min + lat_max) / 2.0
    res_lat = resolution_m / 111000.0
    res_lon = resolution_m / (111000.0 * np.cos(np.radians(lat_c)))
    lon_axis = np.arange(lon_min, lon_max + res_lon, res_lon)
    lat_axis = np.arange(lat_min, lat_max + res_lat, res_lat)
    mesh_lon, mesh_lat = np.meshgrid(lon_axis, lat_axis)
    transform = from_origin(lon_axis.min(), lat_axis.max(), res_lon, res_lat)

    # Grid by true (lon, lat); NaN-valued pixels keep gaps where there is no
    # data (interpolation does not cross them).
    grid = griddata((lon, lat), val, (mesh_lon, mesh_lat), method=method)
    grid = np.flipud(grid).astype(np.float32)
    filled = np.isfinite(grid)
    grid[~filled] = nodata

    nrow, ncol = grid.shape
    src_profile = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=nrow,
        width=ncol,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    )
    dst_profile = cog_profiles.get("deflate")
    with MemoryFile() as mem:
        with mem.open(**src_profile) as src:
            src.write(grid, 1)
        with mem.open() as src:
            cog_translate(
                src,
                out_tif,
                dst_profile,
                overview_resampling="nearest",
                quiet=True,
            )

    is_valid, errors, warnings = cog_validate(out_tif)
    status = "valid" if is_valid else "INVALID"
    print(
        f"COG {status}: {out_tif} "
        f"({int(filled.sum())} cells @ {res_lon:.4f}x{res_lat:.4f} deg)"
    )
    if errors:
        print("  errors:", errors)
    if warnings:
        print("  warnings:", warnings)
    return out_tif


def parse_acquisition_date(path):
    """Parse the acquisition date (YYYYMMDD) from an EMIT/ACOLITE filename.

    Handles both the EMIT L1B granule format
    (``EMIT_L1B_RAD_001_20250414T200042_2510413_035.nc``) and the ACOLITE
    L2W output format (``ISS_EMIT_2025_04_14_20_00_48_L2W.nc``).

    Args:
        path (str): Path to an EMIT L1B or ACOLITE L2W NetCDF file.

    Returns:
        str: The 8-digit date string (e.g. ``"20250414"``).

    Raises:
        ValueError: If no date can be parsed from the filename.
    """
    name = os.path.basename(path)
    match = re.search(r"(\d{8})T\d{6}", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2}", name)
    if match:
        return "".join(match.group(1, 2, 3))
    raise ValueError(f"Could not parse acquisition date from: {path}")


def infer_scene_maps(nc_path, models):
    """Run inference on one EMIT L2W scene and return product maps in memory.

    No files are written. The per-pixel model outputs are reshaped to the
    scene's native swath grid so they can be gridded directly to GeoTIFFs.

    Args:
        nc_path (str): Path to the ACOLITE L2W NetCDF file (with ``Rrs_*``).
        models (dict): Loaded models/scalers from :func:`load_models`.

    Returns:
        dict: ``{"latitude", "longitude", "chla", "tss", "acdom440",
            "valid"}`` where the first five are 2D arrays and ``valid`` is the
            number of valid (finite) chl-a retrieval pixels.
    """
    # Chl-a uses the row-wise min-max model; returns a flat [lat, lon, value].
    chla_flat = preprocess_and_infer_emit_minmax(
        nc_path=nc_path,
        model=models["chla"]["model"],
        full_band_wavelengths=PRODUCT_BANDS["chla"],
        use_spectral_mask=True,
    )

    # TSS / aCDOM use the robust model; return 2D maps plus the geolocation.
    tss_flat, tss_2d, _, lat, lon = preprocess_and_infer_emit_robust(
        nc_path=nc_path,
        model=models["tss"]["model"],
        scaler_Rrs=models["tss"]["scaler_Rrs"],
        TSS_scalers_dict=models["tss"]["scaler_dict"],
        full_band_wavelengths=PRODUCT_BANDS["tss"],
        use_diff=False,
        use_spectral_mask=True,
    )
    acdom_flat, acdom_2d, _, _, _ = preprocess_and_infer_emit_robust(
        nc_path=nc_path,
        model=models["acdom440"]["model"],
        scaler_Rrs=models["acdom440"]["scaler_Rrs"],
        TSS_scalers_dict=models["acdom440"]["scaler_dict"],
        full_band_wavelengths=PRODUCT_BANDS["acdom440"],
        use_diff=False,
        use_spectral_mask=True,
    )

    lat = np.ma.filled(np.asarray(lat), np.nan).astype(np.float64)
    lon = np.ma.filled(np.asarray(lon), np.nan).astype(np.float64)
    shape = lat.shape

    chla = chla_flat[:, 2].reshape(shape).astype(np.float32)
    tss = np.asarray(tss_2d, dtype=np.float32)
    acdom = np.asarray(acdom_2d, dtype=np.float32)

    return {
        "latitude": lat,
        "longitude": lon,
        "chla": chla,
        "tss": tss,
        "acdom440": acdom,
        "valid": int(np.isfinite(chla).sum()),
    }


def scene_stem(input_nc):
    """Return the L1B granule name without its extension.

    The stem is reused verbatim as the COG filename, so an output can always
    be traced back to the exact granule it came from. The EMIT L1B name is
    used (not the ACOLITE L2W name) because it carries the granule ID and the
    ``YYYYMMDDTHHMMSS`` stamp the catalogs group on.

    Args:
        input_nc (str): Path to the EMIT L1B NetCDF file, e.g.
            ``EMIT_L1B_RAD_001_20250414T200042_2510413_035.nc``.

    Returns:
        str: The filename without directory or extension, e.g.
            ``"EMIT_L1B_RAD_001_20250414T200042_2510413_035"``.
    """
    return os.path.splitext(os.path.basename(input_nc))[0]


def scene_cog_paths(save_dir, stem):
    """Build the COG output path for every product of one scene.

    Args:
        save_dir (str): Root output directory.
        stem (str): Granule stem from :func:`scene_stem`.

    Returns:
        dict: Mapping of product label (``chla``/``tss``/``acdom``) to the
            path ``<save_dir>/<label>/<stem>.tif``.
    """
    return {
        label: os.path.join(save_dir, label, f"{stem}.tif")
        for label in PRODUCT_LABELS.values()
    }


def write_scene_cogs(maps, save_dir, stem):
    """Write the in-memory product maps to granule-named gridded COGs.

    Each product goes to ``<save_dir>/<label>/<stem>.tif``, so every pass of a
    given date produces its own set of files instead of overwriting the others.

    Args:
        maps (dict): Output of :func:`infer_scene_maps`.
        save_dir (str): Root output directory.
        stem (str): Granule stem from :func:`scene_stem`.

    Returns:
        list[str]: Paths to the written COGs.
    """
    out_paths = scene_cog_paths(save_dir, stem)
    paths = []
    for var, label in PRODUCT_LABELS.items():
        out_tif = out_paths[label]
        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        paths.append(
            save_product_to_cog(
                out_tif=out_tif,
                lat_2d=maps["latitude"],
                lon_2d=maps["longitude"],
                values_2d=maps[var],
            )
        )
    return paths


def save_products_to_nc(maps, output_nc):
    """Write the merged chl-a / TSS / aCDOM maps to a multi-variable NetCDF.

    The variables share the L2W swath geolocation (2D ``lat``/``lon``), so the
    output preserves the native scene geometry.

    Args:
        maps (dict): Output of :func:`infer_scene_maps`.
        output_nc (str): Output NetCDF path.

    Returns:
        str: The path to the written NetCDF.
    """
    import xarray as xr

    os.makedirs(os.path.dirname(output_nc) or ".", exist_ok=True)
    dims = ("y", "x")
    ds_out = xr.Dataset(
        data_vars={
            "chla": (
                dims,
                maps["chla"],
                {"long_name": "chlorophyll-a concentration", "units": "mg m-3"},
            ),
            "tss": (
                dims,
                maps["tss"],
                {"long_name": "total suspended solids", "units": "g m-3"},
            ),
            "acdom": (
                dims,
                maps["acdom440"],
                {
                    "long_name": "CDOM absorption coefficient at 440 nm",
                    "units": "m-1",
                },
            ),
            "lat": (
                dims,
                maps["latitude"],
                {"long_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                dims,
                maps["longitude"],
                {"long_name": "longitude", "units": "degrees_east"},
            ),
        },
        attrs={
            "title": "EMIT derived water quality products",
            "columns": "lat, lon, chla, tss, acdom",
        },
    )
    encoding = {
        "chla": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        "tss": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        "acdom": {"zlib": True, "complevel": 4, "_FillValue": np.nan},
        "lat": {"zlib": True, "complevel": 4},
        "lon": {"zlib": True, "complevel": 4},
    }
    ds_out.to_netcdf(output_nc, encoding=encoding)
    ds_out.close()
    print("Saved NetCDF:", output_nc)
    return output_nc


def process_scene(
    input_nc,
    models,
    save_dir,
    l2_dir,
    acolite_dir=None,
    download=True,
    write_nc=False,
    reuse_l2w=True,
):
    """Run the full EMIT pipeline on one L1B radiance scene.

    Steps: ACOLITE atmospheric correction (L1B -> L2W) -> MoE-VAE inference ->
    write gridded Cloud Optimized GeoTIFFs (and optionally a merged NetCDF).

    Args:
        input_nc (str): Path to the EMIT ``L1B_RAD`` NetCDF file.
        models (dict): Loaded models/scalers from :func:`load_models`.
        save_dir (str): Directory to write the COG/NetCDF products into.
        l2_dir (str): Root directory for intermediate ACOLITE L2 output.
        acolite_dir (str, optional): ACOLITE install directory (defaults to
            ``ACOLITE_DIR``; downloaded automatically if missing).
        download (bool): Download ACOLITE if not already installed (default
            True).
        write_nc (bool): Also write the merged products NetCDF alongside the
            COGs (default False). The COGs are the published product; the
            NetCDF is a local convenience for keeping the native swath
            geometry.
        reuse_l2w (bool): Reuse an existing ACOLITE L2W product for this scene
            instead of re-running the correction (default True).

    Returns:
        list[str]: Paths to the written COG files.
    """
    print(f"Processing scene: {input_nc}")
    result = run_acolite(
        input_nc,
        l2_dir,
        acolite_dir=acolite_dir,
        download=download,
        reuse=reuse_l2w,
    )
    l2w_path = result["l2w_files"][0]

    maps = infer_scene_maps(l2w_path, models)
    stem = scene_stem(input_nc)
    cogs = write_scene_cogs(maps, save_dir, stem)
    if write_nc:
        save_products_to_nc(maps, os.path.join(save_dir, "nc", f"{stem}.nc"))
    return cogs
