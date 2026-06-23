"""Download the latest EMIT L1B radiance scene(s) for the workflow.

Uses HyperCoast's ``search_emit`` / ``download_emit`` (which wrap NASA
``earthaccess``) to fetch the most recent ``EMITL1BRAD`` granule(s) over a
region of interest into the data folder. The granules include the at-sensor
radiance (``RAD``) and geolocation (``OBS``) files that ACOLITE needs.

For downloading scenes over a specific date range instead, use
``download_data.py``.

A NASA Earthdata login is required. Credentials are read from ``~/.netrc``
(or the ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` environment
variables). Create a free account at https://urs.earthdata.nasa.gov if needed.
"""

import os
import re
from datetime import datetime, timedelta, timezone

import earthaccess
import hypercoast

# Default to the shared data drive; fall back to a local folder if unmounted.
DATA_DIR = os.environ.get("EMIT_DATA_DIR", "/media/hdd/Data/EMIT/data")
os.makedirs(DATA_DIR, exist_ok=True)

# === Search parameters ===
# Bounding box over the Gulf of Mexico / U.S. Gulf coast, matching the region
# covered by the bundled test scene. [xmin, ymin, xmax, ymax]
BBOX = (-98.0, 18.0, -80.0, 31.0)
SHORT_NAME = "EMITL1BRAD"  # at-sensor radiance + geolocation (OBS)
NUM_SCENES = 1  # number of most-recent scenes to download
# Look-back windows (days) tried in order until granules are found. EMIT does
# not image continuously, so a region may not be revisited for weeks.
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


# === Authenticate with NASA Earthdata ===
# Reads credentials from ~/.netrc or EARTHDATA_USERNAME/PASSWORD env vars.
earthaccess.login(persist=True)

# === Search recent windows until granules are found ===
end = datetime.now(timezone.utc)
granules = []
for days in LOOKBACK_DAYS:
    start = end - timedelta(days=days)
    temporal = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(f"Searching {temporal[0]} to {temporal[1]} ...")
    granules = hypercoast.search_emit(
        bbox=BBOX,
        temporal=temporal,
        count=-1,
        short_name=SHORT_NAME,
    )
    if granules:
        break

if not granules:
    raise RuntimeError(
        f"No EMIT {SHORT_NAME} granules found over the bounding box in the "
        f"last {LOOKBACK_DAYS[-1]} days."
    )

# Keep the latest scenes by acquisition time.
granules = sorted(granules, key=acquisition_time, reverse=True)[:NUM_SCENES]

print(f"\nLatest {len(granules)} scene(s):")
for g in granules:
    print("  ", g.get("meta", {}).get("native-id", g))

# === Download (each granule includes the RAD and OBS files) ===
files = hypercoast.download_emit(granules, out_dir=DATA_DIR)
print(f"\nDownloaded {len(files)} file(s) to {DATA_DIR}:")
for f in files:
    print("  ", os.path.basename(f))
