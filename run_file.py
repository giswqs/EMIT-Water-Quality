"""Process a single EMIT L1B radiance scene into water-quality products.

Runs ACOLITE atmospheric correction (L1B -> L2W) followed by MoE-VAE
inference on one EMIT scene and writes validated Cloud Optimized GeoTIFFs
(chl-a, TSS, aCDOM). Each COG keeps the granule name of the input and goes in
a per-product subfolder::

    output/chla/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
    output/tss/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif
    output/acdom/EMIT_L1B_RAD_001_20250414T200042_2510413_035.tif

Examples::

    python run_file.py EMIT_L1B_RAD_001_20250414T200042_2510413_035.nc
    python run_file.py /path/to/EMIT_L1B_RAD_...nc --output /path/to/output

ACOLITE must be installed locally; point at it via the ``ACOLITE_DIR``
environment variable (or ``--acolite`` / ``--proj-dir``). The matching
``L1B_OBS`` geolocation granule must sit next to the ``RAD`` file.

To process every scene in a folder, use ``run_folder.py``.
"""

import os
import argparse

import torch

from emit_processing import BASE_DIR, load_models, process_scene

# Default to the shared data drive; fall back to local folders next to the
# scripts if the drive is not mounted.
DEFAULT_ROOT = "/media/hdd/Data/EMIT"
DEFAULT_DATA = os.path.join(DEFAULT_ROOT, "data")
DEFAULT_OUTPUT = os.path.join(DEFAULT_ROOT, "output")
DEFAULT_L2 = os.path.join(DEFAULT_ROOT, "L2")

parser = argparse.ArgumentParser(
    description="Process a single EMIT L1B radiance scene into water-quality "
    "products."
)
parser.add_argument(
    "input",
    help="Input EMIT L1B_RAD NetCDF file. Either a path, or a filename in the "
    "data folder.",
)
parser.add_argument(
    "--model-dir",
    default=os.path.join(BASE_DIR, "model"),
    help="Directory containing the model subfolders (default: ./model).",
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
    "--write-nc",
    action="store_true",
    help="Also write a merged per-scene NetCDF to <output>/nc (default: off).",
)
args = parser.parse_args()

# Resolve the input path: use it as given if it exists, otherwise look in the
# default data folder.
if os.path.isfile(args.input):
    input_nc = os.path.abspath(args.input)
else:
    input_nc = os.path.join(DEFAULT_DATA, args.input)
if not os.path.isfile(input_nc):
    raise FileNotFoundError(f"Input file not found: {args.input}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

models = load_models(args.model_dir, device)
process_scene(
    input_nc,
    models,
    save_dir=args.output,
    l2_dir=args.l2_dir,
    acolite_dir=args.acolite_dir,
    download=not args.no_download,
    write_nc=args.write_nc,
)
print("Done.")
