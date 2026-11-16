"""Process EMIT L1B radiance scenes in a folder into daily water-quality COGs.

For each acquisition date, every EMIT L1B pass in the folder is atmospherically
corrected with ACOLITE and run through the models; the pass with the most valid
retrieval pixels is kept and written as date-named Cloud Optimized GeoTIFFs
(chl-a, TSS, aCDOM) plus a merged products NetCDF in the output folder.

Examples::

    python run_folder.py                       # process the default data dir
    python run_folder.py /path/to/scenes --output /path/to/output

ACOLITE must be installed locally; point at it via the ``ACOLITE_DIR``
environment variable (or ``--acolite`` / ``--proj-dir``). Only ``L1B_RAD``
granules are processed; the matching ``L1B_OBS`` granules must sit alongside
them.

To process a single file, use ``run_file.py``.
"""

import argparse
import glob
import os
from collections import defaultdict

import torch

from emit_processing import (
    BASE_DIR,
    infer_scene_maps,
    load_models,
    parse_acquisition_date,
    run_acolite,
    save_products_to_nc,
    write_scene_cogs,
)

DEFAULT_ROOT = "/media/hdd/Data/EMIT"
DEFAULT_DATA = os.path.join(DEFAULT_ROOT, "data")
DEFAULT_OUTPUT = os.path.join(DEFAULT_ROOT, "output")
DEFAULT_L2 = os.path.join(DEFAULT_ROOT, "L2")

parser = argparse.ArgumentParser(
    description="Process EMIT L1B radiance scenes in a folder into daily "
    "water-quality COGs (best pass per day)."
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
args = parser.parse_args()

if not os.path.isdir(args.folder):
    raise NotADirectoryError(f"Input folder not found: {args.folder}")

# Group input scenes by acquisition date.
by_date = defaultdict(list)
for path in sorted(glob.glob(os.path.join(args.folder, args.pattern))):
    by_date[parse_acquisition_date(path)].append(path)

if not by_date:
    raise FileNotFoundError(
        f"No files matching '{args.pattern}' found in {args.folder}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
n_scenes = sum(len(v) for v in by_date.values())
print(f"Found {n_scenes} scene(s) across {len(by_date)} date(s) in {args.folder}")

models = load_models(args.model_dir, device)

succeeded, failed = [], []
for date in sorted(by_date):
    passes = by_date[date]
    print(f"\n[{date}] {len(passes)} pass(es)")
    best_maps, best_pass = None, None
    for input_nc in passes:
        try:
            result = run_acolite(
                input_nc,
                args.l2_dir,
                acolite_dir=args.acolite_dir,
                download=not args.no_download,
            )
            maps = infer_scene_maps(result["l2w_files"][0], models)
        except Exception as exc:  # noqa: BLE001 - keep batch going on failure
            print(f"  FAILED {os.path.basename(input_nc)}: {exc}")
            failed.append((input_nc, exc))
            continue
        print(f"  {os.path.basename(input_nc)}: {maps['valid']} valid pixels")
        if best_maps is None or maps["valid"] > best_maps["valid"]:
            best_maps, best_pass = maps, input_nc

    if best_maps is None:
        continue
    print(f"  -> best: {os.path.basename(best_pass)} ({best_maps['valid']} px)")
    write_scene_cogs(best_maps, args.output, date)
    save_products_to_nc(
        best_maps, os.path.join(args.output, f"EMIT-{date}-products.nc")
    )
    succeeded.append(date)

print(f"\nDone. {len(succeeded)} date(s) written, {len(failed)} pass(es) failed.")
for input_nc, exc in failed:
    print(f"  - {os.path.basename(input_nc)}: {exc}")
