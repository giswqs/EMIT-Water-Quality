"""Download EMIT L1B radiance scenes over a specified date range.

Uses HyperCoast's ``search_emit`` / ``download_emit`` (which wrap NASA
``earthaccess``) to fetch granules over a region of interest and date range
into the data folder. The granules match the ``EMITL1BRAD`` product (at-sensor
radiance plus the ``OBS`` geolocation granule), which ACOLITE then
atmospherically corrects before inference.

For downloading only the most recent scene instead, use ``download_latest.py``.

Examples::

    python download_data.py 2025-04-01 2025-04-30
    python download_data.py 2025-04-01 2025-04-30 --count 5
    python download_data.py 2025-04-14 2025-04-14 --bbox -99 18 -78 42

A NASA Earthdata login is required. Credentials are read from ``~/.netrc``
(or the ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` environment
variables). Create a free account at https://urs.earthdata.nasa.gov if needed.
"""

import os
import argparse

import earthaccess
import hypercoast

# Default to the shared data drive; fall back to a local folder if unmounted.
DEFAULT_DATA_DIR = os.environ.get("EMIT_DATA_DIR", "/media/hdd/Data/EMIT/data")

# Default bounding box over the Gulf of Mexico / U.S. Gulf coast, matching the
# region covered by the bundled test scene. [xmin, ymin, xmax, ymax]
DEFAULT_BBOX = (-98.0, 18.0, -80.0, 31.0)
SHORT_NAME = "EMITL1BRAD"  # at-sensor radiance + geolocation (OBS)

# === Command-line arguments ===
parser = argparse.ArgumentParser(
    description="Download EMIT L1B radiance scenes over a specified date range."
)
parser.add_argument("start", help="Start date (YYYY-MM-DD).")
parser.add_argument("end", help="End date (YYYY-MM-DD), inclusive.")
parser.add_argument(
    "--count",
    type=int,
    default=3,
    help="Maximum number of granules to download (default: 3). Use -1 for all.",
)
parser.add_argument(
    "--bbox",
    type=float,
    nargs=4,
    metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    default=DEFAULT_BBOX,
    help="Bounding box (default: Gulf of Mexico).",
)
parser.add_argument(
    "--short-name",
    default=SHORT_NAME,
    help=f"EMIT dataset short name (default: {SHORT_NAME}).",
)
parser.add_argument(
    "--out-dir",
    default=DEFAULT_DATA_DIR,
    help=f"Directory to download into (default: {DEFAULT_DATA_DIR}).",
)
args = parser.parse_args()
DATA_DIR = args.out_dir
os.makedirs(DATA_DIR, exist_ok=True)

# === Authenticate with NASA Earthdata ===
# Reads credentials from ~/.netrc or EARTHDATA_USERNAME/PASSWORD env vars.
earthaccess.login(persist=True)

# === Search ===
print(f"Searching {args.start} to {args.end} over {tuple(args.bbox)} ...")
granules = hypercoast.search_emit(
    bbox=tuple(args.bbox),
    temporal=(args.start, args.end),
    count=-1,
    short_name=args.short_name,
)

# Trim to the requested count (-1 keeps all).
if args.count != -1:
    granules = granules[: args.count]

if not granules:
    raise RuntimeError(
        f"No EMIT {args.short_name} granules found over {tuple(args.bbox)} "
        f"between {args.start} and {args.end}."
    )

print(f"Found {len(granules)} granule(s).")

# === Download (each granule includes the RAD and OBS files) ===
files = hypercoast.download_emit(granules, out_dir=DATA_DIR)
print(f"\nDownloaded {len(files)} file(s) to {DATA_DIR}:")
for f in files:
    print("  ", os.path.basename(f))
