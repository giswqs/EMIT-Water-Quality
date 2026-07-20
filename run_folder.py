"""Process EMIT L1B radiance scenes in a folder into per-scene water-quality COGs.

Every EMIT L1B pass in the folder is atmospherically corrected with ACOLITE,
run through the models, and written as validated Cloud Optimized GeoTIFFs
(chl-a, TSS, aCDOM). Each COG keeps the granule name of its input and lives in
a per-product subfolder, so dates with several passes keep every pass::

    output/chla/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
    output/tss/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
    output/acdom/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif

Products are gridded from the swath onto a regular EPSG:4326 grid. Scenes
whose COGs already exist are skipped unless ``--overwrite`` is passed, so an
interrupted backfill can simply be re-run — which matters here because ACOLITE
is the slow step.

Examples::

    python run_folder.py                       # process the default data dir
    python run_folder.py /path/to/scenes --output /path/to/output
    python run_folder.py --json-dir /media/hdd/Data/EMIT/json

ACOLITE must be installed locally; point at it via the ``ACOLITE_DIR``
environment variable (or ``--acolite-dir``). Only ``L1B_RAD`` granules are
processed; the matching ``L1B_OBS`` granules must sit alongside them.

To process a single file, use ``run_file.py``.
"""

import os
import glob
import argparse

import torch

from emit_processing import (
    BASE_DIR,
    load_models,
    run_acolite,
    infer_scene_maps,
    write_scene_cogs,
    save_products_to_nc,
    scene_stem,
    scene_cog_paths,
    parse_acquisition_date,
)

DEFAULT_ROOT = "/media/hdd/Data/EMIT"
DEFAULT_DATA = os.path.join(DEFAULT_ROOT, "data")
DEFAULT_OUTPUT = os.path.join(DEFAULT_ROOT, "output")
DEFAULT_L2 = os.path.join(DEFAULT_ROOT, "L2")

parser = argparse.ArgumentParser(
    description="Process every EMIT L1B radiance scene in a folder into "
    "per-scene water-quality COGs."
)
parser.add_argument(
    "folder",
    nargs="?",
    default=DEFAULT_DATA,
    help=f"Folder containing EMIT L1B_RAD NetCDF files (default: {DEFAULT_DATA}).",
)
parser.add_argument(
    "--output",
    default=DEFAULT_OUTPUT,
    help=f"Output directory for the products (default: {DEFAULT_OUTPUT}).",
)
parser.add_argument(
    "--l2-dir",
    default=DEFAULT_L2,
    help=f"Directory for intermediate ACOLITE L2 output (default: {DEFAULT_L2}).",
)
parser.add_argument(
    "--model-dir",
    default=os.path.join(BASE_DIR, "model"),
    help="Directory containing the model subfolders (default: ./model).",
)
parser.add_argument(
    "--pattern",
    default="*L1B_RAD*.nc",
    help="Glob pattern for input files (default: *L1B_RAD*.nc).",
)
parser.add_argument(
    "--acolite-dir",
    default=None,
    help="ACOLITE install directory (default: $ACOLITE_DIR; downloaded "
    "automatically if missing).",
)
parser.add_argument(
    "--no-download",
    action="store_true",
    help="Do not download ACOLITE if it is missing (error instead).",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Reprocess scenes whose COGs already exist (default: skip them).",
)
parser.add_argument(
    "--write-nc",
    action="store_true",
    help="Also write a merged per-scene NetCDF to <output>/nc (default: off).",
)
parser.add_argument(
    "--json-dir",
    default=None,
    help="If set, write the per-date GeoJSON catalogs here after processing.",
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Process at most this many scenes (useful for a quick test run).",
)
args = parser.parse_args()

if not os.path.isdir(args.folder):
    raise NotADirectoryError(f"Input folder not found: {args.folder}")

scenes = sorted(glob.glob(os.path.join(args.folder, args.pattern)))

if not scenes:
    raise FileNotFoundError(
        f"No files matching '{args.pattern}' found in {args.folder}"
    )

# Skip scenes that already have a complete set of COGs.
if not args.overwrite:
    pending = [
        path
        for path in scenes
        if not all(
            os.path.isfile(p)
            for p in scene_cog_paths(args.output, scene_stem(path)).values()
        )
    ]
    n_skipped = len(scenes) - len(pending)
    if n_skipped:
        print(f"Skipping {n_skipped} scene(s) that already have COGs.")
    scenes = pending

if args.limit is not None:
    scenes = scenes[: args.limit]

if not scenes:
    print("Nothing to process.")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dates = {parse_acquisition_date(p) for p in scenes}
    print(f"Processing {len(scenes)} scene(s) across {len(dates)} date(s)")

    models = load_models(args.model_dir, device)

    succeeded, failed = [], []
    for i, input_nc in enumerate(scenes, start=1):
        name = os.path.basename(input_nc)
        print(f"\n[{i}/{len(scenes)}] {name}")
        try:
            result = run_acolite(
                input_nc,
                args.l2_dir,
                acolite_dir=args.acolite_dir,
                download=not args.no_download,
            )
            maps = infer_scene_maps(result["l2w_files"][0], models)
            print(f"  {maps['valid']} valid pixels")
            stem = scene_stem(input_nc)
            write_scene_cogs(maps, args.output, stem)
            if args.write_nc:
                save_products_to_nc(maps, os.path.join(args.output, "nc", f"{stem}.nc"))
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            print(f"  FAILED: {exc}")
            failed.append((input_nc, exc))
            continue
        succeeded.append(input_nc)

    print(f"\nDone. {len(succeeded)} scene(s) written, {len(failed)} failed.")
    for input_nc, exc in failed:
        print(f"  - {os.path.basename(input_nc)}: {exc}")

if args.json_dir:
    from make_json import build_catalogs

    build_catalogs(args.output, args.json_dir)
