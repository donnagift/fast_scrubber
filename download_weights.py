#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the fast_scrubber model weights from OSF (https://osf.io/x95g7).

Usage:
    python download_weights.py                       # all 5 folds into ./weights
    python download_weights.py --dest /opt/weights   # elsewhere
    python download_weights.py --folds 0             # one fold only (e.g. for CI)

Only the files inference needs are fetched (dataset.json, plans.json and
fold_*/checkpoint_final.pth, ~1.2 GB for all folds), and each is checked
against a pinned SHA-256. Files already present with the right hash are kept.
The layout written under --dest is the one fastscrubber.py expects:

    <dest>/fast_scrubber/nnUNet_results/Dataset500_Segmentation/
        nnUNetTrainer__nnUNetPlans__3d_fullres/{dataset.json,plans.json,fold_*/}

Only the Python standard library is used.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path


# --- Internal constants ------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()

OSF_PROJECT = "x95g7"
OSF_API     = f"https://api.osf.io/v2/nodes/{OSF_PROJECT}/files/osfstorage/"
MODEL_PATH  = ("fast_scrubber", "nnUNet_results", "Dataset500_Segmentation",
               "nnUNetTrainer__nnUNetPlans__3d_fullres")
ALL_FOLDS   = ["0", "1", "2", "3", "4"]

# The checkpoints and plans.json are byte-identical to the deployment IronFist
# runs (resources/provenance/fastscrubber-deployment.json there). OSF's
# dataset.json differs from that deployment's in metadata only (description,
# creators, channel name); inference reads its labels and file ending.
SHA256 = {
    "dataset.json":                "f051f58130a7bc0bf76cd997456088cb1f6ae043180ff55090ff4a36e94ecdc5",
    "plans.json":                  "28912f1e245355941b332f87efffc871f79d4e0bf334795bb3d536e68f04dc5f",
    "fold_0/checkpoint_final.pth": "66fa8cfe8399acd017d78597c65ce06061d12beea3debb4bc809313d5b1eb1dc",
    "fold_1/checkpoint_final.pth": "d088889c1407fd44296f6900d0242036b2f1c51a0d42f47004f712223eea762b",
    "fold_2/checkpoint_final.pth": "f94185fe1a5df96d5996b4e7bb21e0c96e4706c9e4076e4e474f8e81468b1604",
    "fold_3/checkpoint_final.pth": "5b40c796b95858e8ea51a29327832acaddff5fa5ef9532d3bfb7b1fb44edaef5",
    "fold_4/checkpoint_final.pth": "011fe28c72e4bb12c50f4c1b3252f84a45ad615b57fbc6c6383a66dec55d5043",
}

RETRIES = 3


# --- Helpers -----------------------------------------------------------------

def get_json(url: str) -> dict:
    """GET an OSF API page, retrying transient network errors."""
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except OSError as error:
            if attempt == RETRIES:
                raise
            print(f"[WARN] {url}: {error}; retrying", flush=True)
            time.sleep(5 * attempt)


def list_folder(url: str) -> dict:
    """Map each entry name of an OSF folder listing to its API record."""
    entries = {}
    while url:
        page = get_json(url)
        for item in page["data"]:
            entries[item["attributes"]["name"]] = item
        url = page["links"].get("next")
    return entries


def children_url(item: dict) -> str:
    return item["relationships"]["files"]["links"]["related"]["href"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected: str) -> None:
    """Download to a .part file, verify its hash, then move it into place."""
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            break
        except OSError as error:
            if attempt == RETRIES:
                raise
            print(f"[WARN] {url}: {error}; retrying", flush=True)
            time.sleep(5 * attempt)

    actual = sha256(partial)
    if actual != expected:
        partial.unlink()
        sys.exit(
            f"[ERROR] SHA-256 mismatch for {dest.name}: got {actual}, expected "
            f"{expected}. The weights on OSF may have changed."
        )
    partial.replace(dest)


# --- Main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the fast_scrubber model weights from OSF."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=SCRIPT_DIR / "weights",
        help=f"Destination directory (default: {SCRIPT_DIR / 'weights'}).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=ALL_FOLDS,
        default=ALL_FOLDS,
        help="Folds to download (default: all 5).",
    )
    args = parser.parse_args()

    model_dir = args.dest.resolve().joinpath(*MODEL_PATH)
    wanted = ["dataset.json", "plans.json"]
    wanted += [f"fold_{fold}/checkpoint_final.pth" for fold in args.folds]

    pending = []
    for relative in wanted:
        dest = model_dir / relative
        if dest.is_file() and sha256(dest) == SHA256[relative]:
            print(f"[INFO] Up to date: {dest}", flush=True)
        else:
            pending.append(relative)

    if pending:
        print(f"[INFO] Locating weights in OSF project {OSF_PROJECT}", flush=True)
        folder = list_folder(OSF_API)
        for name in MODEL_PATH:
            folder = list_folder(children_url(folder[name]))

        for relative in pending:
            entries = folder
            *parents, name = relative.split("/")
            for parent in parents:
                entries = list_folder(children_url(entries[parent]))
            item = entries[name]

            dest = model_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            size_mb = (item["attributes"]["size"] or 0) / 1e6
            print(f"[INFO] Downloading {relative} ({size_mb:.0f} MB)", flush=True)
            download(item["links"]["download"], dest, SHA256[relative])

    print(f"\n[DONE] Weights ready in {model_dir}\n", flush=True)


if __name__ == "__main__":
    main()
