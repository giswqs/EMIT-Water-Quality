"""Build the per-date GeoJSON catalogs for the published EMIT COGs.

For every acquisition date and product, one JSON file is written listing all
the COGs from that date, following the same minimal FeatureCollection shape
as https://data.source.coop/giswqs/opengeos/naip_nd_2023_stac.json::

    {"type": "FeatureCollection",
     "features": [{"bbox": [...], "assets": {"image": {"href": "..."}}}]}

Each feature is one EMIT pass, its ``bbox`` read from the COG and its ``href``
pointing at the published copy of that COG. EMIT images in narrow (~75 km)
swaths and crosses a region several times a day, so a date usually produces
several features — together they mosaic into that day's coverage.

The catalogs are derived from the COGs on disk, so this can be re-run at any
time (after a backfill, or after new scenes are added) without reprocessing.

Examples::

    python make_json.py                               # ./output -> ./json
    python make_json.py /media/hdd/Data/EMIT/output --json-dir /media/hdd/Data/EMIT/json
    python make_json.py --base-url https://example.com/data
"""

import os
import re
import glob
import json
import argparse
from collections import defaultdict

import rasterio

from emit_processing import BASE_DIR, PRODUCT_LABELS, HF_DATA_URL


def cog_bbox(tif_path):
    """Read a COG's bounding box in EPSG:4326.

    Args:
        tif_path (str): Path to the GeoTIFF.

    Returns:
        list[float]: ``[west, south, east, north]`` in degrees.
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
    return [
        round(bounds.left, 6),
        round(bounds.bottom, 6),
        round(bounds.right, 6),
        round(bounds.top, 6),
    ]


def build_catalogs(output_dir, json_dir, base_url=HF_DATA_URL):
    """Write one GeoJSON catalog per acquisition date and product.

    Args:
        output_dir (str): Root directory holding the ``<product>/*.tif`` COGs.
        json_dir (str): Directory to write the ``<date>_<product>.json`` files
            into. Created if missing.
        base_url (str): Base URL the COGs are published under. Hrefs are built
            as ``<base_url>/<product>/<granule>.tif``.

    Returns:
        list[str]: Paths to the JSON files that were written.
    """
    os.makedirs(json_dir, exist_ok=True)
    base_url = base_url.rstrip("/")
    written = []

    for label in sorted(set(PRODUCT_LABELS.values())):
        product_dir = os.path.join(output_dir, label)
        if not os.path.isdir(product_dir):
            print(f"No {label} directory in {output_dir}; skipping.")
            continue

        # Group this product's COGs by acquisition date.
        by_date = defaultdict(list)
        for tif in sorted(glob.glob(os.path.join(product_dir, "*.tif"))):
            match = re.search(r"(\d{8})T\d{6}", os.path.basename(tif))
            if match is None:
                print(f"  Skipping (no date in name): {os.path.basename(tif)}")
                continue
            by_date[match.group(1)].append(tif)

        for date, tifs in sorted(by_date.items()):
            features = []
            for tif in tifs:
                name = os.path.basename(tif)
                features.append(
                    {
                        "bbox": cog_bbox(tif),
                        "assets": {"image": {"href": f"{base_url}/{label}/{name}"}},
                    }
                )
            out_json = os.path.join(json_dir, f"{date}_{label}.json")
            # Compact (no indent/spaces), matching the reference catalog.
            with open(out_json, "w") as f:
                json.dump(
                    {"type": "FeatureCollection", "features": features},
                    f,
                    separators=(",", ":"),
                )
            print(f"{out_json}: {len(features)} scene(s)")
            written.append(out_json)

    print(f"\nWrote {len(written)} JSON file(s) to {json_dir}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the per-date GeoJSON catalogs for the EMIT COGs."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=os.path.join(BASE_DIR, "output"),
        help="Directory holding the <product>/*.tif COGs (default: ./output).",
    )
    parser.add_argument(
        "--json-dir",
        default=os.path.join(BASE_DIR, "json"),
        help="Directory to write the JSON catalogs into (default: ./json).",
    )
    parser.add_argument(
        "--base-url",
        default=HF_DATA_URL,
        help=f"Base URL the COGs are published under (default: {HF_DATA_URL}).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise NotADirectoryError(f"Output folder not found: {args.output_dir}")

    build_catalogs(args.output_dir, args.json_dir, args.base_url)
