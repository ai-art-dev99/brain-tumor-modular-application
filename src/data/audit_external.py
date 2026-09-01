#!/usr/bin/env python3
"""
audit_external.py -- prove a candidate external dataset does not overlap the
development data, before any model touches it.

WHY THIS RUNS FIRST
-------------------
An external test set is only external if none of its images appear in
training. Publishers routinely describe their datasets as independent while
redistributing images from the same upstream repositories: this project has
already established that SARTAJ is largely a re-encoding of the Figshare
collection, which nobody had reported. The claim must therefore be checked,
not accepted, and checked with the same dual perceptual hash calibrated
earlier -- exact hashing cannot survive the re-encoding that redistribution
involves.

The comparison is made against the RAW indices, not the deduplicated
development set. An external image matching something that was dropped during
deduplication is still an image the backbone saw during fine-tuning, since
fine-tuning used representatives drawn from those clusters.

PSEUDO-PATIENT GROUPING
-----------------------
PMRAM, like SARTAJ and BR35H, ships no patient identifiers. Multiple slices of
one patient will therefore be treated as independent observations unless they
are grouped, and confidence intervals computed over ungrouped slices are too
narrow -- the same defect this project corrected internally. Near-duplicate
clustering is applied to the external set as well, and the resulting clusters
are used as the resampling unit. They are pseudo-patients and must be
described as such.

WHAT IT WRITES
--------------
  external_manifest.csv   one row per external image, with class, hashes,
                          pseudo-group, and whether it is a representative
  overlap_report.json     match counts at several thresholds, worked examples,
                          and the verdict

Usage
-----
    python audit_external.py --name pmram --root /workspace/data/raw/pmram
    python audit_external.py --name pmram --root ... --montage 12
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import hashlib
import json
import re
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

MANIFEST = Path("/workspace/data/manifest")
EXTERNAL = Path("/workspace/data/external")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_ALIASES = {
    "glioma": "glioma", "brain_glioma": "glioma", "glioma_tumor": "glioma",
    "meningioma": "meningioma", "brain_menin": "meningioma",
    "meningioma_tumor": "meningioma",
    "pituitary": "pituitary", "brain_tumor": "pituitary",
    "pituitary_tumor": "pituitary",
    "notumor": "notumor", "no_tumor": "notumor", "normal": "notumor",
    "brain_normal": "notumor", "no": "notumor", "healthy": "notumor",
}

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def sha256_of(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def hex_to_packed(s: pd.Series) -> np.ndarray:
    return np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in s],
                    dtype=np.uint8)


def dual_distance(pa, da, pb, db, chunk: int = 256):
    """Best dual-hash match of each row of A against all of B.

    Returns (best_index, best_dual_distance) where the dual distance is the
    larger of the two Hamming distances, so a pair counts as close only if it
    is close under both hash families.
    """
    best_i = np.zeros(len(pa), dtype=np.int64)
    best_d = np.full(len(pa), 255, dtype=np.int16)
    for s in range(0, len(pa), chunk):
        e = min(s + chunk, len(pa))
        dp = POPCOUNT[pa[s:e, None, :] ^ pb[None, :, :]].sum(axis=2)
        dd = POPCOUNT[da[s:e, None, :] ^ db[None, :, :]].sum(axis=2)
        d = np.maximum(dp, dd).astype(np.int16)
        best_i[s:e] = d.argmin(axis=1)
        best_d[s:e] = d.min(axis=1)
    return best_i, best_d


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def normalise_class(path: Path) -> str:
    for part in reversed(path.parts):
        k = part.strip().lower().replace(" ", "_").replace("-", "_")
        # Some redistributions prefix class folders with the image size, e.g.
        # PMRAM's "512Glioma" / "512Normal". Strip a leading run of digits
        # before matching; without this every image falls through to "unknown"
        # and the class counts, grouping and per-class overlap figures are all
        # computed on nothing.
        k = re.sub(r"^\d+[_\-]?", "", k)
        if k in CLASS_ALIASES:
            return CLASS_ALIASES[k]
    return "unknown"


def index_external(root: Path) -> pd.DataFrame:
    paths = [Path(p) for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise SystemExit(f"no images under {root}")
    print(f"==> indexing {len(paths)} images under {root}")
    rows = []
    for p in tqdm(paths, unit="img"):
        try:
            with Image.open(p) as im:
                w, h = im.size
                g = im.convert("L")
                ph, dh = imagehash.phash(g, 8), imagehash.dhash(g, 8)
        except Exception as e:
            print(f"    unreadable, skipped: {p} ({e})")
            continue
        rows.append({"path": str(p), "rel": str(p.relative_to(root)),
                     "label": normalise_class(p.relative_to(root)),
                     "width": w, "height": h, "bytes": p.stat().st_size,
                     "sha256": sha256_of(p), "phash": str(ph), "dhash": str(dh)})
    return pd.DataFrame(rows)


def load_development() -> pd.DataFrame:
    """Every development image, before deduplication.

    The backbone was fine-tuned on representatives of clusters that also
    contained the images dropped as duplicates, so an external match to a
    dropped image is still contamination.
    """
    frames = []
    f = pd.read_csv(MANIFEST / "files_index.csv")
    frames.append(f[["path", "source", "class", "phash", "dhash", "sha256"]])
    fig = pd.read_csv(MANIFEST / "figshare_index.csv")
    frames.append(pd.DataFrame({"path": fig.render_path, "source": "figshare",
                                "class": fig["class"], "phash": fig.phash,
                                "dhash": fig.dhash, "sha256": ""}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="short id, e.g. pmram")
    ap.add_argument("--root", required=True)
    ap.add_argument("--dedup-threshold", type=int, default=2,
                    help="internal near-duplicate threshold; keep at the value "
                         "calibrated on the development data")
    ap.add_argument("--expect-per-class", type=int, default=None,
                    help="raw count per class; a mismatch usually means the "
                         "augmented copy was downloaded")
    ap.add_argument("--montage", type=int, default=8)
    a = ap.parse_args()

    EXTERNAL.mkdir(parents=True, exist_ok=True)
    ext = index_external(Path(a.root))

    print("\n  class counts:")
    print(ext.label.value_counts().to_string())
    if (ext.label == "unknown").any():
        print(f"\n  WARNING: {(ext.label == 'unknown').sum()} images did not map "
              f"to a class.\n  Folders seen: "
              f"{sorted({Path(r).parts[0] for r in ext.rel})}")

    if a.expect_per_class:
        bad = ext.label.value_counts()[lambda s: s != a.expect_per_class]
        if len(bad):
            print(f"\n  !! class counts differ from the expected "
                  f"{a.expect_per_class} per class:")
            print(bad.to_string())
            print("  For PMRAM the raw set is 400 per class; 1500 per class is")
            print("  the augmented copy. Evaluating on augmented images would")
            print("  repeat the defect this project set out to correct.")

    dims = (ext.width.astype(str) + "x" + ext.height.astype(str))
    print(f"\n  distinct dimensions: {dims.nunique()} "
          f"(most common {dims.value_counts().index[0]})")

    # -- internal duplicates ---------------------------------------------------
    dup_exact = ext[ext.duplicated("sha256", keep=False)]
    print(f"\n  byte-identical duplicates inside the external set: "
          f"{len(dup_exact)} files in {dup_exact.sha256.nunique()} groups")

    ph, dh = hex_to_packed(ext.phash), hex_to_packed(ext.dhash)
    dsu = DSU(len(ext))
    n_edges = 0
    for s in range(0, len(ext), 256):
        e = min(s + 256, len(ext))
        dp = POPCOUNT[ph[s:e, None, :] ^ ph[None, :, :]].sum(axis=2)
        dd = POPCOUNT[dh[s:e, None, :] ^ dh[None, :, :]].sum(axis=2)
        r, c = np.where((dp <= a.dedup_threshold) & (dd <= a.dedup_threshold))
        for i, j in zip(r, c):
            gi, gj = s + int(i), int(j)
            if gi < gj:
                dsu.union(gi, gj)
                n_edges += 1
    ext["group_id"] = [f"{a.name}_g{dsu.find(i):05d}" for i in range(len(ext))]
    order = ext.sort_values(["group_id", "bytes"], ascending=[True, False])
    ext["is_representative"] = ext.index.isin(
        set(order.drop_duplicates("group_id", keep="first").index))

    print(f"  near-duplicate edges: {n_edges}")
    print(f"  pseudo-patient groups: {ext.group_id.nunique()} "
          f"({len(ext)} images, {len(ext) / ext.group_id.nunique():.2f} per group)")
    print(f"  representatives: {int(ext.is_representative.sum())}")

    # -- overlap with development ---------------------------------------------
    dev = load_development()
    print(f"\n==> cross-checking against {len(dev)} development images "
          f"(pre-deduplication)")
    dph, ddh = hex_to_packed(dev.phash), hex_to_packed(dev.dhash)
    bi, bd = dual_distance(ph, dh, dph, ddh)

    ext["nearest_dev_path"] = dev.path.to_numpy()[bi]
    ext["nearest_dev_source"] = dev.source.to_numpy()[bi]
    ext["nearest_dev_class"] = dev["class"].to_numpy()[bi]
    ext["nearest_dev_distance"] = bd

    exact = set(ext.sha256) & set(dev.sha256[dev.sha256 != ""])
    print(f"\n  byte-identical matches: {len(exact)}")

    print("\n  nearest-neighbour dual-hash distance distribution:")
    hist = pd.Series(bd).value_counts().sort_index()
    mx = hist.max()
    for d, n in hist.head(24).items():
        print(f"    {d:3d} | {n:5d} {'#' * min(60, int(60 * n / mx))}")

    counts = {}
    for t in (0, 2, 4, 6, 8, 10):
        counts[t] = int((bd <= t).sum())
        print(f"  matches at dual-hash <= {t:2d}: {counts[t]:5d} "
              f"({100 * counts[t] / len(ext):.2f}%)")

    verdict_t = a.dedup_threshold
    n_hit = counts[verdict_t]
    print()
    if len(exact) == 0 and n_hit == 0:
        print(f"  VERDICT: no overlap detected at the calibrated threshold "
              f"({verdict_t}).")
        print(f"  The nearest external image is {bd.min()} bits from anything in")
        print(f"  the development data. This dataset can serve as an external")
        print(f"  test set. State the threshold and this distribution in the")
        print(f"  manuscript so the check is auditable.")
    else:
        print(f"  VERDICT: {n_hit} external images match development images at")
        print(f"  dual-hash <= {verdict_t}, of which {len(exact)} are byte-identical.")
        print(f"  This dataset is NOT independent as distributed. Either exclude")
        print(f"  the matching images and report how many were removed, or drop")
        print(f"  the dataset. Do not evaluate on it unfiltered.")
        ex = ext[bd <= verdict_t].head(10)
        print("\n  examples:")
        print(ex[["rel", "label", "nearest_dev_source", "nearest_dev_class",
                  "nearest_dev_distance"]].to_string(index=False))

    ext["overlaps_development"] = bd <= verdict_t

    if a.montage:
        write_montages(ext, a.name, a.montage)

    out = EXTERNAL / f"{a.name}_manifest.csv"
    ext.to_csv(out, index=False)
    report = {
        "name": a.name, "root": str(a.root), "n_images": len(ext),
        "class_counts": ext.label.value_counts().to_dict(),
        "distinct_dimensions": int(dims.nunique()),
        "internal_exact_duplicate_files": int(len(dup_exact)),
        "pseudo_groups": int(ext.group_id.nunique()),
        "representatives": int(ext.is_representative.sum()),
        "dedup_threshold": a.dedup_threshold,
        "development_images_compared": len(dev),
        "byte_identical_matches": len(exact),
        "matches_by_threshold": counts,
        "min_distance": int(bd.min()),
        "overlap_detected": bool(len(exact) or n_hit),
    }
    (EXTERNAL / f"{a.name}_overlap_report.json").write_text(
        json.dumps(report, indent=2))
    print(f"\n  wrote {out}")
    print(f"  wrote {EXTERNAL / f'{a.name}_overlap_report.json'}")


def write_montages(ext: pd.DataFrame, name: str, n: int) -> None:
    """Closest external-to-development pairs, side by side. A hash distance is
    a number; whether two images are the same scan is a judgement, and it
    should be made by looking."""
    d = EXTERNAL / f"{name}_closest_pairs"
    d.mkdir(parents=True, exist_ok=True)
    T = 240
    closest = ext.nsmallest(n, "nearest_dev_distance")
    for rank, (_, r) in enumerate(closest.iterrows(), 1):
        sheet = Image.new("RGB", (T * 2, T + 24), "black")
        for k, p in enumerate([r.path, r.nearest_dev_path]):
            try:
                sheet.paste(Image.open(p).convert("RGB").resize((T, T)),
                            (k * T, 24))
            except Exception:
                pass
        sheet.save(d / f"pair{rank:02d}_d{int(r.nearest_dev_distance)}"
                      f"_{r.label}_vs_{r.nearest_dev_source}.png")
    print(f"\n  wrote {n} closest-pair montages to {d}")
    print("  Left: external. Right: nearest development image. If any pair is")
    print("  visibly the same scan, the numeric verdict above is wrong and the")
    print("  threshold needs revisiting.")


if __name__ == "__main__":
    main()