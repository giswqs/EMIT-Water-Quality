"""Download the most recent EMIT L1B radiance scenes for the workflow.

Uses HyperCoast's ``search_emit`` / ``download_emit`` (which wrap NASA
``earthaccess``) to fetch ``EMITL1BRAD`` granules over a region of interest
into the data folder. The granules include the at-sensor radiance (``RAD``)
and geolocation (``OBS``) files that ACOLITE needs.

Granules are grouped by acquisition date and the most recent ``--dates`` dates
are kept, taking at most ``--passes-per-date`` passes from each. EMIT images in
narrow (~75 km) swaths and crosses a region several times a day, so capping the
passes keeps a backfill to a predictable size — each pass is a ~1.9 GB download
and its own ACOLITE run.

Examples::

    python download_latest.py                        # newest date, 1 pass
    python download_latest.py --dates 5 --passes-per-date 2
    python download_latest.py --dates 5 --bbox -99 18 -78 42

For downloading scenes over a specific date range instead, use
``download_data.py``.

A NASA Earthdata login is required. Credentials are read from ``~/.netrc``
(or the ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` environment
variables). Create a free account at https://urs.earthdata.nasa.gov if needed.
"""

import os
import re
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import earthaccess
import hypercoast

# Default to the shared data drive; fall back to a local folder if unmounted.
DEFAULT_DATA_DIR = os.environ.get("EMIT_DATA_DIR", "/media/hdd/Data/EMIT/data")

# Default bounding box over the Gulf of Mexico / U.S. Gulf coast, matching the
# region covered by the bundled test scene. [xmin, ymin, xmax, ymax]
DEFAULT_BBOX = (-98.0, 18.0, -80.0, 31.0)
SHORT_NAME = "EMITL1BRAD"  # at-sensor radiance + geolocation (OBS)
# Look-back windows (days) tried in order until enough dates are found. EMIT
# does not image continuously, so a region may not be revisited for weeks.
LOOKBACK_DAYS = (7, 30, 90, 365)


def acquisition_time(granule):
    """Extract the acquisition timestamp (YYYYMMDDTHHMMSS) from a granule.

    Args:
        granule (dict): An earthaccess granule result.

    Returns:
        str: The timestamp string, or "" if it cannot be parsed. The string
            sorts chronologically, so it doubles as a sort key.
    """
    native_id = granule.get("meta", {}).get("native-id", "")
    match = re.search(r"(\d{8}T\d{6})", native_id)
    return match.group(1) if match else ""


def select_latest(granules, num_dates, passes_per_date):
    """Keep the newest ``num_dates`` acquisition dates, capped per date.

    Args:
        granules (list): Granule results from ``hypercoast.search_emit``.
        num_dates (int): Number of most-recent acquisition dates to keep.
        passes_per_date (int): Maximum passes to keep from each date. Use -1
            to keep every pass.

    Returns:
        list: The selected granules, newest first.
    """
    by_date = defaultdict(list)
    for granule in granules:
        stamp = acquisition_time(granule)
        if stamp:
            by_date[stamp[:8]].append((stamp, granule))

    selected = []
    for date in sorted(by_date, reverse=True)[:num_dates]:
        passes = [g for _, g in sorted(by_date[date], key=lambda p: p[0], reverse=True)]
        if passes_per_date != -1:
            passes = passes[:passes_per_date]
        selected.extend(passes)
    return selected


parser = argparse.ArgumentParser(
    description="Download the most recent EMIT L1B radiance scenes."
)
parser.add_argument(
    "--dates",
    type=int,
    default=1,
    help="Number of most-recent acquisition dates to download (default: 1).",
)
parser.add_argument(
    "--passes-per-date",
    type=int,
    default=1,
    help="Maximum passes to keep per date (default: 1). Use -1 for all.",
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
os.makedirs(args.out_dir, exist_ok=True)

# === Authenticate with NASA Earthdata ===
# Reads credentials from ~/.netrc or EARTHDATA_USERNAME/PASSWORD env vars.
earthaccess.login(persist=True)

# === Search widening windows until enough dates are found ===
end = datetime.now(timezone.utc)
selected = []
for days in LOOKBACK_DAYS:
    start = end - timedelta(days=days)
    temporal = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(f"Searching {temporal[0]} to {temporal[1]} ...")
    granules = hypercoast.search_emit(
        bbox=tuple(args.bbox),
        temporal=temporal,
        count=-1,
        short_name=args.short_name,
    )
    selected = select_latest(granules, args.dates, args.passes_per_date)
    # Widen the window until the requested number of dates is covered, so a
    # sparsely imaged region still yields a full backfill.
    if len({acquisition_time(g)[:8] for g in selected}) >= args.dates:
        break

if not selected:
    raise RuntimeError(
        f"No EMIT {args.short_name} granules found over {tuple(args.bbox)} in "
        f"the last {LOOKBACK_DAYS[-1]} days."
    )

dates = sorted({acquisition_time(g)[:8] for g in selected}, reverse=True)
print(f"\n{len(selected)} pass(es) across {len(dates)} date(s):")
for g in selected:
    print("  ", g.get("meta", {}).get("native-id", g))

# === Download (each granule includes the RAD and OBS files) ===
files = hypercoast.download_emit(selected, out_dir=args.out_dir)
print(f"\nDownloaded {len(files)} file(s) to {args.out_dir}:")
for f in files:
    print("  ", os.path.basename(f))
