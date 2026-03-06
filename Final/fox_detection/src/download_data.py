"""
download_data.py
Download fox and non-fox audio recordings from Xeno-canto API v3.

Usage:
    python -m src.download_data --xc_key YOUR_API_KEY

Fox queries (Vulpes vulpes, quality A + B):
    gen:vulpes+sp:vulpes+q:A
    gen:vulpes+sp:vulpes+q:B

Non-fox negative queries:
    grp:birds+cnt:europe+q:A      (European birds)
    grp:"land mammals"+q:A        (other land mammals, excluding Vulpes)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import requests


API_URL = "https://xeno-canto.org/api/3/recordings"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _query_xc(
    query: str,
    api_key: str,
) -> list[dict]:
    """Run a paginated Xeno-canto query and return all recording dicts."""
    recordings: list[dict] = []
    page = 1

    while True:
        params = {"query": query, "key": api_key, "page": page}
        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("recordings", [])
        recordings.extend(batch)

        num_pages = int(data.get("numPages", 1))
        print(f"  Page {page}/{num_pages} – {len(batch)} recordings")

        if page >= num_pages:
            break
        page += 1
        time.sleep(0.5)

    return recordings


def _download_file(url: str, dest: str) -> bool:
    """Download a single file.  Returns True on success."""
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"    ⚠ Failed to download {url}: {exc}")
        return False


def _ensure_url(file_url: str) -> str:
    """Prepend https: if the URL starts with //."""
    if file_url.startswith("//"):
        return "https:" + file_url
    return file_url


# ── Main logic ────────────────────────────────────────────────────────────────

def download_data(
    xc_key: str,
    fox_dir: str = "data/raw/fox",
    nonfox_dir: str = "data/raw/nonfox",
) -> tuple[int, int]:
    """Download fox and non-fox recordings from Xeno-canto.

    Returns
    -------
    tuple[int, int]
        (fox_downloaded, nonfox_downloaded) counts of *newly* downloaded files.
    """
    os.makedirs(fox_dir, exist_ok=True)
    os.makedirs(nonfox_dir, exist_ok=True)

    # ── Fox recordings ────────────────────────────────────────────────────
    fox_queries = [
        "gen:vulpes sp:vulpes q:A",
        "gen:vulpes sp:vulpes q:B",
    ]
    fox_recordings: list[dict] = []
    for q in fox_queries:
        print(f"\n🔍 Fox query: {q}")
        fox_recordings.extend(_query_xc(q, xc_key))

    fox_downloaded = 0
    fox_skipped = 0
    for rec in fox_recordings:
        xc_id = rec.get("id", "unknown")
        dest = os.path.join(fox_dir, f"XC{xc_id}.mp3")
        if os.path.exists(dest):
            fox_skipped += 1
            continue
        file_url = _ensure_url(rec.get("file", ""))
        if not file_url:
            continue
        print(f"  ⬇ XC{xc_id}.mp3")
        if _download_file(file_url, dest):
            fox_downloaded += 1
        time.sleep(0.5)

    print(f"\n🦊 Fox: {fox_downloaded} downloaded, {fox_skipped} skipped (already exist)")

    # ── Non-fox recordings ────────────────────────────────────────────────
    # Use common European species that are well-represented on Xeno-canto.
    # The grp: field is not supported in API v3, so we query by genus/species.
    nonfox_queries = [
        "gen:turdus sp:merula q:A cnt:germany",       # Common blackbird
        "gen:erithacus sp:rubecula q:A cnt:germany",   # European robin
        "gen:parus sp:major q:A cnt:germany",          # Great tit
        "gen:fringilla sp:coelebs q:A cnt:germany",    # Common chaffinch
        "gen:corvus sp:corone q:A cnt:germany",        # Carrion crow
        "gen:canis sp:lupus q:A",                      # Grey wolf
        "gen:cervus sp:elaphus q:A",                   # Red deer
    ]
    nonfox_recordings: list[dict] = []
    for q in nonfox_queries:
        print(f"\n🔍 Non-fox query: {q}")
        try:
            recs = _query_xc(q, xc_key)
        except Exception as exc:
            print(f"    ⚠ Query failed ({exc}), skipping.")
            continue
        # Exclude any Vulpes recordings
        recs = [r for r in recs if r.get("gen", "").lower() != "vulpes"]
        nonfox_recordings.extend(recs)

    nonfox_downloaded = 0
    nonfox_skipped = 0
    for rec in nonfox_recordings:
        xc_id = rec.get("id", "unknown")
        dest = os.path.join(nonfox_dir, f"XC{xc_id}.mp3")
        if os.path.exists(dest):
            nonfox_skipped += 1
            continue
        file_url = _ensure_url(rec.get("file", ""))
        if not file_url:
            continue
        print(f"  ⬇ XC{xc_id}.mp3")
        if _download_file(file_url, dest):
            nonfox_downloaded += 1
        time.sleep(0.5)

    print(f"\n🐦 Non-fox: {nonfox_downloaded} downloaded, {nonfox_skipped} skipped (already exist)")

    # ── Summary ───────────────────────────────────────────────────────────
    total_fox = len([f for f in os.listdir(fox_dir) if f.endswith(".mp3")])
    total_nonfox = len([f for f in os.listdir(nonfox_dir) if f.endswith(".mp3")])
    print(f"\n{'='*50}")
    print(f"Summary: {total_fox} fox files, {total_nonfox} non-fox files downloaded")
    print(f"{'='*50}")

    return fox_downloaded, nonfox_downloaded


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download fox & non-fox audio from Xeno-canto API v3."
    )
    parser.add_argument(
        "--xc_key", type=str, required=True,
        help="Xeno-canto API key.",
    )
    parser.add_argument(
        "--fox_dir", type=str, default="data/raw/fox",
        help="Output directory for fox recordings (default: data/raw/fox).",
    )
    parser.add_argument(
        "--nonfox_dir", type=str, default="data/raw/nonfox",
        help="Output directory for non-fox recordings (default: data/raw/nonfox).",
    )
    args = parser.parse_args()

    download_data(
        xc_key=args.xc_key,
        fox_dir=args.fox_dir,
        nonfox_dir=args.nonfox_dir,
    )
