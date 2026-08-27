#!/usr/bin/env python3
"""
build_manifest.py -- construct the provenance manifest for the composite
brain tumour MRI dataset.

WHAT THIS SOLVES
----------------
The working dataset (the aggregated Kaggle redistribution) has had its
provenance stripped: filenames were rewritten, patient identifiers dropped,
tumour masks discarded, and the originating repository forgotten. Reviewer
points 2, 3, 5 and 9 are all unanswerable in that state.

This script rebuilds the missing links by indexing every source, then
matching each working image back to its origin perceptually.

STAGES
------
  figshare : read the 3,064 original .mat files, recover PID / label /
             tumourMask, render an 8-bit view, and hash it.
  files    : index every JPEG/PNG in SARTAJ, BR35H and the aggregated set
             (path, class, dimensions, sha256, pHash, dHash).
  match    : link each aggregated image to its source repository, and where
             that source is Figshare, attach the patient ID and mask path.
  all      : run the three stages in order.

OUTPUT
------
  data/manifest/figshare_index.csv
  data/manifest/files_index.csv
  data/manifest/manifest.csv          <- the artefact everything downstream reads
  data/manifest/match_distances.csv   <- evidence for the chosen threshold

Usage
-----
    python build_manifest.py --stage all
    python build_manifest.py --stage match --phash-threshold 8
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from pathlib import Path

import h5py
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path("/workspace/data")
RAW = ROOT / "raw"
CLEAN = ROOT / "clean"
MANIFEST = ROOT / "manifest"

FIGSHARE_MAT = RAW / "figshare" / "mat"
SOURCE_ROOTS = {
    "sartaj": RAW / "sartaj",
    "br35h": RAW / "br35h",
    "aggregated": RAW / "aggregated",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Cheng's label encoding. NOTE: this is NOT the same ordering that
# torchvision's ImageFolder produces (which sorts alphabetically:
# glioma=0, meningioma=1, notumor=2, pituitary=3). Conflating the two
# silently swaps glioma and meningioma through the entire analysis.
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}

# Directories that are not part of the classification task. The Mask-RCNN
# subset is a byte-identical copy of BR35H's `yes` folder repackaged for
# object detection; `pred` is an unlabelled inference folder. Including
# either inflates counts and injects duplicates.
EXCLUDE_DIR_PARTS = {"br35h-mask-rcnn", "pred"}

# Folder names vary across repositories; normalise to one vocabulary.
CLASS_ALIASES = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "glioma_tumour": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "meningioma_tumour": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "pituitary_tumour": "pituitary",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no_tumour": "notumor",
    "no": "notumor",
    # BR35H's "yes" folder contains tumours of unspecified type. The aggregated
    # dataset drew only the "no" folder from BR35H. Labelling these as a
    # distinct class keeps them from silently contaminating a tumour class.
    "yes": "tumour_unspecified",
    "pred": "unlabelled",
}

# Lookup table for vectorised Hamming distance over packed hash bytes.
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# =============================================================================
# Helpers
# =============================================================================

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Exact content hash. Detects byte-identical duplicates."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Render a 16-bit MRI slice as 8-bit greyscale.

    Min-max scaling is used deliberately: it is the transform that public
    redistributions of this dataset almost always apply, so matching against
    them succeeds. It is monotonic and linear, which also means the perceptual
    hash is largely invariant to the exact choice.
    """
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    return (((img - lo) / (hi - lo)) * 255.0).round().astype(np.uint8)


def hashes_of(pil_img: Image.Image) -> tuple[str, str, np.ndarray]:
    """
    Return (phash_hex, dhash_hex, packed_phash_bytes).

    Two hash families are computed because they fail differently: pHash is a
    DCT of a downsampled image and is robust to brightness and mild rescaling;
    dHash keys on horizontal gradients and is a useful tiebreaker when several
    slices from the same patient collide under pHash.
    """
    grey = pil_img.convert("L")
    ph = imagehash.phash(grey, hash_size=8)
    dh = imagehash.dhash(grey, hash_size=8)
    packed = np.packbits(ph.hash.flatten())
    return str(ph), str(dh), packed


def hamming_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pairwise Hamming distance between packed hashes.
    a: (n, 8) uint8, b: (m, 8) uint8 -> (n, m) uint8
    """
    return POPCOUNT[a[:, None, :] ^ b[None, :, :]].sum(axis=2)


def normalise_class(path: Path, source: str) -> str:
    """Infer the class label from the directory name, normalised."""
    for part in reversed(path.parts):
        key = part.strip().lower().replace(" ", "_").replace("-", "_")
        if key in CLASS_ALIASES:
            return CLASS_ALIASES[key]
    return "unknown"


def infer_split(path: Path) -> str:
    """Record the redistribution's own train/test assignment, for reference only."""
    parts = {p.lower() for p in path.parts}
    if {"training", "train"} & parts:
        return "train"
    if {"testing", "test"} & parts:
        return "test"
    return "unassigned"


# =============================================================================
# Stage 1: Figshare
# =============================================================================

def stage_figshare(render: bool = True) -> pd.DataFrame:
    """
    Read every original .mat file and recover the metadata that every
    redistribution discards.
    """
    files = sorted(glob.glob(str(FIGSHARE_MAT / "*.mat")))
    if not files:
        sys.exit(f"ERROR: no .mat files under {FIGSHARE_MAT}. Run fetch_figshare.sh first.")

    print(f"==> Reading {len(files)} Figshare .mat files")
    img_dir = CLEAN / "figshare" / "images"
    msk_dir = CLEAN / "figshare" / "masks"
    if render:
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

    rows, packed_hashes = [], []

    for path in tqdm(files, unit="file"):
        stem = Path(path).stem
        with h5py.File(path, "r") as h:
            d = h["cjdata"]
            # PID is a MATLAB char array; h5py hands back uint16 codepoints.
            pid = "".join(chr(c) for c in np.array(d["PID"]).ravel()).strip()
            label = int(np.array(d["label"]).ravel()[0])
            # HDF5 is row-major, MATLAB is column-major: transpose or the mask
            # will not align with the image, silently corrupting every IoU
            # figure computed later.
            image = np.array(d["image"]).T
            mask = np.array(d["tumorMask"]).T.astype(np.uint8)

        img8 = to_uint8(image)
        pil = Image.fromarray(img8)
        ph, dh, packed = hashes_of(pil)
        packed_hashes.append(packed)

        img_path = msk_path = ""
        if render:
            img_path = str(img_dir / f"{stem}.png")
            msk_path = str(msk_dir / f"{stem}.png")
            pil.save(img_path, optimize=True)
            Image.fromarray(mask * 255).save(msk_path, optimize=True)

        rows.append({
            "mat_file": stem,
            "patient_id": f"figshare_{pid}",
            "label_code": label,
            "class": FIGSHARE_LABELS.get(label, "unknown"),
            "height": image.shape[0],
            "width": image.shape[1],
            "dtype": str(image.dtype),
            "intensity_min": int(image.min()),
            "intensity_max": int(image.max()),
            "tumour_px": int(mask.sum()),
            "tumour_frac": float(mask.sum() / mask.size),
            "render_path": img_path,
            "mask_path": msk_path,
            "phash": ph,
            "dhash": dh,
        })

    df = pd.DataFrame(rows)
    np.save(MANIFEST / "figshare_phash.npy", np.vstack(packed_hashes))
    MANIFEST.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST / "figshare_index.csv", index=False)

    # --- the numbers that answer reviewer point 2 for this subset ------------
    print(f"\n  slices           : {len(df)}")
    print(f"  unique patients  : {df.patient_id.nunique()}")
    print(f"  class counts     :\n{df['class'].value_counts().to_string()}")
    print("\n  slices per patient:")
    spp = df.groupby("patient_id").size()
    print(f"    min {spp.min()}  median {spp.median():.0f}  "
          f"mean {spp.mean():.1f}  max {spp.max()}")
    print("\n  patients per class:")
    print(df.groupby("class").patient_id.nunique().to_string())

    # This is the crux of reviewer point 3: many slices per patient means an
    # image-level random split places near-identical slices of the same brain
    # on both sides of the train/test boundary.
    print(f"\n  NOTE: {spp.mean():.1f} slices per patient on average. Any split "
          f"not grouped by\n        patient_id will leak.")
    return df


# =============================================================================
# Stage 2: index loose image files
# =============================================================================

def stage_files() -> pd.DataFrame:
    """Index every image in SARTAJ, BR35H and the aggregated redistribution."""
    rows, packed_hashes = [], []

    for source, root in SOURCE_ROOTS.items():
        if not root.exists():
            print(f"  WARNING: {root} not found, skipping '{source}'")
            continue

        paths = [Path(p) for p in glob.glob(str(root / "**" / "*"), recursive=True)
                 if Path(p).suffix.lower() in IMAGE_EXTS]
        paths = [p for p in paths
                 if not (EXCLUDE_DIR_PARTS & {q.lower() for q in p.parts})]
        print(f"==> Indexing {len(paths)} images from '{source}'")

        for p in tqdm(paths, unit="img"):
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    mode = im.mode
                    ph, dh, packed = hashes_of(im)
            except Exception as e:
                print(f"    unreadable, skipped: {p} ({e})")
                continue

            packed_hashes.append(packed)
            rows.append({
                "path": str(p),
                "source": source,
                "class": normalise_class(p.relative_to(root), source),
                "orig_split": infer_split(p.relative_to(root)),
                "width": w,
                "height": h,
                "mode": mode,
                "bytes": p.stat().st_size,
                "sha256": sha256_of(p),
                "phash": ph,
                "dhash": dh,
            })

    df = pd.DataFrame(rows)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    np.save(MANIFEST / "files_phash.npy", np.vstack(packed_hashes))
    df.to_csv(MANIFEST / "files_index.csv", index=False)

    print("\n  images per source and class:")
    print(pd.crosstab(df["source"], df["class"]).to_string())

    # Contradicts the manuscript's claim that all images were originally
    # 512x512: only the Figshare subset is.
    print("\n  image dimensions per source:")
    dims = df.assign(dim=df.width.astype(str) + "x" + df.height.astype(str))
    for src, g in dims.groupby("source"):
        top = g.dim.value_counts().head(4)
        print(f"    {src}: {g.dim.nunique()} distinct sizes; most common -> "
              + ", ".join(f"{k} ({v})" for k, v in top.items()))

    # Free byte-identical duplicate check, courtesy of sha256.
    dup = df[df.duplicated("sha256", keep=False)]
    print(f"\n  byte-identical duplicates: {len(dup)} files in "
          f"{dup.sha256.nunique()} groups")
    cross = dup.groupby("sha256").source.nunique()
    print(f"  of which span more than one source: {(cross > 1).sum()} groups")
    return df


# =============================================================================
# Stage 3: match working images back to their origin
# =============================================================================

def stage_match(threshold: int = 8) -> pd.DataFrame:
    """
    Link each aggregated image to its originating repository.

    Exact hashing cannot be used: the redistribution re-encoded 16-bit .mat
    slices as 8-bit JPEG, so no byte-level identity survives. Perceptual
    hashing is the appropriate tool.
    """
    fig = pd.read_csv(MANIFEST / "figshare_index.csv")
    files = pd.read_csv(MANIFEST / "files_index.csv")
    fig_h = np.load(MANIFEST / "figshare_phash.npy")
    files_h = np.load(MANIFEST / "files_phash.npy")

    agg_mask = (files.source == "aggregated").to_numpy()
    agg = files[agg_mask].reset_index(drop=True)
    agg_h = files_h[agg_mask]
    print(f"==> Matching {len(agg)} aggregated images against "
          f"{len(fig)} Figshare slices")

    best_d = np.empty(len(agg), dtype=np.int16)
    best_i = np.empty(len(agg), dtype=np.int64)
    second_d = np.empty(len(agg), dtype=np.int16)

    CHUNK = 256
    for s in tqdm(range(0, len(agg), CHUNK), unit="chunk"):
        e = min(s + CHUNK, len(agg))
        D = hamming_block(agg_h[s:e], fig_h).astype(np.int16)
        order = np.argsort(D, axis=1)
        best_i[s:e] = order[:, 0]
        best_d[s:e] = np.take_along_axis(D, order[:, :1], axis=1).ravel()
        second_d[s:e] = np.take_along_axis(D, order[:, 1:2], axis=1).ravel()

    agg["best_distance"] = best_d
    agg["runner_up_distance"] = second_d
    # A small gap between the best and second-best match means the two
    # candidates are adjacent slices of one patient. The patient assignment is
    # then still correct even if the exact slice is ambiguous.
    agg["match_margin"] = second_d - best_d
    agg["matched_mat"] = fig.mat_file.to_numpy()[best_i]
    agg["matched_patient"] = fig.patient_id.to_numpy()[best_i]
    agg["matched_class"] = fig["class"].to_numpy()[best_i]
    agg["matched_mask"] = fig.mask_path.to_numpy()[best_i]

    is_fig = agg.best_distance <= threshold
    agg["source_resolved"] = np.where(is_fig, "figshare", "unresolved")
    agg["patient_id"] = np.where(is_fig, agg.matched_patient, "")
    agg["mask_path"] = np.where(is_fig, agg.matched_mask, "")

    # --- threshold evidence, for the manuscript ------------------------------
    hist = pd.Series(best_d).value_counts().sort_index()
    hist.to_csv(MANIFEST / "match_distances.csv", header=["count"])
    print("\n  best-match Hamming distance distribution:")
    for d, n in hist.head(20).items():
        bar = "#" * min(60, int(60 * n / hist.max()))
        print(f"    {d:3d} | {n:6d} {bar}")
    print("    (a clear gap between a low-distance mode and the rest is what "
          "justifies\n     the threshold; report this figure rather than "
          "asserting a value)")

    print(f"\n  resolved to Figshare (d <= {threshold}): "
          f"{int(is_fig.sum())} / {len(agg)} ({100 * is_fig.mean():.1f}%)")

    # Label agreement is an independent check on the matching: if the recovered
    # Figshare label disagrees with the folder the redistribution filed the
    # image under, either the match is wrong or the redistribution mislabelled
    # it. Both are worth knowing about.
    both = agg[is_fig & agg["class"].isin(FIGSHARE_LABELS.values())]
    if len(both):
        agree = (both["class"] == both.matched_class).mean()
        print(f"  class agreement on matched pairs: {100 * agree:.1f}%")
        bad = both[both["class"] != both.matched_class]
        if len(bad):
            print(f"  DISAGREEMENTS ({len(bad)}):")
            print(pd.crosstab(bad["class"], bad.matched_class).to_string())

    resolved = agg[is_fig]
    print(f"\n  distinct patients recovered: {resolved.patient_id.nunique()}")

    # The headline leakage figure: patients whose slices the redistribution
    # placed on both sides of its own train/test boundary.
    if "orig_split" in resolved.columns:
        spl = resolved.groupby("patient_id").orig_split.nunique()
        n_leak = int((spl > 1).sum())
        leaked_imgs = resolved[resolved.patient_id.isin(spl[spl > 1].index)]
        print(f"\n  patients appearing in BOTH the redistribution's train and "
              f"test folders: {n_leak}")
        print(f"  images involved: {len(leaked_imgs)}")
        if n_leak:
            print("  -> the published split is not patient-level; this is the "
                  "direct\n     evidence for reviewer points 3 and 5.")

    out = agg.drop(columns=["matched_patient", "matched_class", "matched_mask"])
    out.to_csv(MANIFEST / "manifest.csv", index=False)
    print(f"\n  wrote {MANIFEST / 'manifest.csv'}")
    return out


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["figshare", "files", "match", "all"],
                    default="all")
    ap.add_argument("--phash-threshold", type=int, default=8,
                    help="max Hamming distance for a match (inspect the "
                         "printed histogram before trusting the default)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip writing PNG renders and masks")
    args = ap.parse_args()

    MANIFEST.mkdir(parents=True, exist_ok=True)

    if args.stage in ("figshare", "all"):
        stage_figshare(render=not args.no_render)
        print()
    if args.stage in ("files", "all"):
        stage_files()
        print()
    if args.stage in ("match", "all"):
        stage_match(threshold=args.phash_threshold)


if __name__ == "__main__":
    main()