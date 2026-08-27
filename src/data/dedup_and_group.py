#!/usr/bin/env python3
"""
dedup_and_group.py -- build the curated dataset and assign leakage-safe groups.

WHY THIS EXISTS
---------------
A model must never be tested on an image that is a near-copy of one it trained
on. Two mechanisms produce such near-copies here:

  (a) the same image redistributed under more than one filename or repository;
  (b) adjacent MRI slices of one patient, which are distinct images but show
      the same brain millimetres apart.

Only (a) is visible to a hash-equality check. (b) is what makes the published
train/test split of this benchmark unsound, and it can only be handled by
grouping images by patient.

This script therefore does three things:
  1. applies explicit inclusion/exclusion rules to build the curated dataset;
  2. removes exact and near-duplicate images, keeping one representative;
  3. assigns every surviving image a group_id that any split must respect.

GROUPING POLICY
---------------
Groups are connected components of a graph whose edges are:
  - shared Figshare patient ID (a true, metadata-backed link), and
  - perceptual-hash distance below a threshold (an inferred link).

Merging is deliberately conservative. If a cluster spans two patient IDs, both
patients are merged into one group rather than split apart. Over-merging costs
a little statistical efficiency; under-merging silently reintroduces the exact
leakage this script exists to remove.

For SARTAJ and BR35H, no patient identifiers exist in any released version of
those repositories. The clusters formed there are pseudo-patient groups: they
mitigate leakage but are NOT equivalent to patient-level grouping, and the
manuscript must say so.

Usage
-----
    python dedup_and_group.py --phash-threshold 6
    python dedup_and_group.py --phash-threshold 6 --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path("/workspace/data/manifest")

# =============================================================================
# Inclusion / exclusion rules -- reviewer point 2 asks for these explicitly,
# so they live in one auditable place rather than being scattered through code.
# =============================================================================

INCLUSION = {
    # class      : list of (source, source_class) pairs to draw from
    "glioma":     [("figshare", "glioma")],
    "meningioma": [("figshare", "meningioma"), ("sartaj", "meningioma")],
    "pituitary":  [("figshare", "pituitary"), ("sartaj", "pituitary")],
    "notumor":    [("br35h", "notumor")],
}

EXCLUSION_RATIONALE = {
    ("sartaj", "glioma"):
        "Documented labelling inconsistency in the SARTAJ glioma folder; "
        "gliomas are drawn from Figshare instead.",
    ("sartaj", "notumor"):
        "500 files reduce to 352 unique hashes (29.6% internally duplicated), "
        "of which 120 are byte-identical to BR35H images.",
    ("br35h", "tumour_unspecified"):
        "BR35H 'yes' images are tumours of unspecified histology and cannot be "
        "assigned to one of the four diagnostic classes.",
    ("br35h", "unknown"):
        "Br35H-Mask-RCNN directory is a byte-identical repackaging of the "
        "'yes' folder for an object-detection task.",
    ("br35h", "unlabelled"):
        "'pred' directory carries no ground-truth labels.",
}


# =============================================================================
# Union-find
# =============================================================================

class DSU:
    """Disjoint-set union with path compression."""

    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def near_duplicate_edges(packed: np.ndarray, threshold: int,
                         chunk: int = 512) -> list[tuple[int, int]]:
    """
    All index pairs whose packed pHashes lie within `threshold` bits.

    Brute force over ~6k images is ~18M comparisons: a few seconds with
    numpy, and exact. An ANN index would be needed only at a much larger
    scale, and would trade away the exactness that makes this auditable.
    """
    n = len(packed)
    edges: list[tuple[int, int]] = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = POPCOUNT[packed[s:e, None, :] ^ packed[None, :, :]].sum(axis=2)
        # Keep the upper triangle only, so each pair is emitted once.
        rows, cols = np.where(d <= threshold)
        for r, c in zip(rows, cols):
            i = s + int(r)
            j = int(c)
            if i < j:
                edges.append((i, j))
    return edges


# =============================================================================

def build(phash_threshold: int, dry_run: bool) -> pd.DataFrame:
    files = pd.read_csv(MANIFEST / "files_index.csv")
    packed_all = np.load(MANIFEST / "files_phash.npy")
    fig = pd.read_csv(MANIFEST / "figshare_index.csv")

    # -- 1. apply inclusion rules ---------------------------------------------
    print("=" * 70)
    print("1. Curating the dataset")
    print("=" * 70)

    keep = np.zeros(len(files), dtype=bool)
    assigned = np.full(len(files), "", dtype=object)
    for cls, pairs in INCLUSION.items():
        for src, src_cls in pairs:
            m = ((files.source == src) & (files["class"] == src_cls)).to_numpy()
            keep |= m
            assigned[m] = cls
            print(f"  {cls:<12} <- {src:<10} {src_cls:<20} {int(m.sum()):5d}")

    print("\n  excluded:")
    for (src, src_cls), why in EXCLUSION_RATIONALE.items():
        n = int(((files.source == src) & (files["class"] == src_cls)).sum())
        if n:
            print(f"    {src}/{src_cls} ({n}): {why}")

    df = files[keep].copy().reset_index(drop=True)
    df["label"] = assigned[keep]
    packed = packed_all[keep]
    print(f"\n  curated total: {len(df)}")
    print(df.label.value_counts().to_string())

    # -- 2. attach Figshare metadata ------------------------------------------
    # Figshare images are indexed from the .mat renders, so the patient ID is
    # a direct join rather than a perceptual match.
    print()
    print("=" * 70)
    print("2. Attaching patient identifiers")
    print("=" * 70)

    fig_by_render = fig.set_index("render_path")
    pid = []
    mask = []
    for p, src in zip(df.path, df.source):
        if src == "figshare" and p in fig_by_render.index:
            pid.append(fig_by_render.at[p, "patient_id"])
            mask.append(fig_by_render.at[p, "mask_path"])
        else:
            pid.append("")
            mask.append("")
    df["patient_id"] = pid
    df["mask_path"] = mask

    has_pid = df.patient_id != ""
    print(f"  images with a true patient ID : {int(has_pid.sum())} "
          f"({100 * has_pid.mean():.1f}%)")
    print(f"  distinct patients             : {df.patient_id[has_pid].nunique()}")
    print(f"  images without any identifier : {int((~has_pid).sum())} "
          f"-> pseudo-patient grouping only")

    # -- 3. build the grouping graph ------------------------------------------
    print()
    print("=" * 70)
    print(f"3. Grouping (pHash threshold = {phash_threshold})")
    print("=" * 70)

    dsu = DSU(len(df))

    # Edge type A: shared patient ID. Metadata-backed, always trusted.
    n_pid_edges = 0
    for _, idx in df[has_pid].groupby("patient_id").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            dsu.union(idx[0], j)
            n_pid_edges += 1
    print(f"  edges from shared patient ID : {n_pid_edges}")

    # Edge type B: perceptual similarity.
    edges = near_duplicate_edges(packed, phash_threshold)
    for i, j in edges:
        dsu.union(i, j)
    print(f"  edges from pHash similarity  : {len(edges)}")

    df["group_id"] = [f"g{dsu.find(i):05d}" for i in range(len(df))]
    n_groups = df.group_id.nunique()
    print(f"\n  resulting groups: {n_groups}")

    sizes = df.groupby("group_id").size()
    print(f"  group size: min {sizes.min()}  median {sizes.median():.0f}  "
          f"mean {sizes.mean():.1f}  max {sizes.max()}")

    # A group that swallowed many patients means the threshold is too loose and
    # the split will be starved of independent units. Surface it rather than
    # letting it distort the folds silently.
    merged = (df[has_pid].groupby("group_id").patient_id.nunique())
    multi = merged[merged > 1]
    if len(multi):
        print(f"\n  groups spanning >1 patient ID: {len(multi)} "
              f"(largest merges {multi.max()} patients)")
        print("  -> conservative: these patients share near-identical images, "
              "so keeping\n     them together is correct. If the largest is "
              "implausible, lower the threshold.")

    # Cross-class groups are a red flag: near-identical images filed under two
    # diagnoses. Report, then let the split keep them together anyway.
    cls_span = df.groupby("group_id").label.nunique()
    n_cls_span = int((cls_span > 1).sum())
    print(f"\n  groups spanning >1 class: {n_cls_span}")
    if n_cls_span:
        bad = df[df.group_id.isin(cls_span[cls_span > 1].index)]
        print(pd.crosstab(bad.group_id, bad.label).sum().to_string())
        bad.to_csv(MANIFEST / "cross_class_groups.csv", index=False)
        print(f"  -> written to {MANIFEST / 'cross_class_groups.csv'} for review")

    # -- 4. select one representative per near-duplicate cluster ---------------
    # Distinct slices of one patient are NOT duplicates and must be kept: they
    # are legitimate independent-ish samples once grouping prevents leakage.
    # Only images that are near-identical to each other are collapsed.
    print()
    print("=" * 70)
    print("4. Removing near-duplicate images")
    print("=" * 70)

    dup_dsu = DSU(len(df))
    for i, j in edges:
        dup_dsu.union(i, j)
    df["dup_cluster"] = [dup_dsu.find(i) for i in range(len(df))]

    # Prefer the largest file in each cluster: least re-compressed, so least
    # information already discarded.
    df = df.sort_values(["dup_cluster", "bytes"], ascending=[True, False])
    df["is_representative"] = ~df.duplicated("dup_cluster", keep="first")

    n_removed = int((~df.is_representative).sum())
    print(f"  near-duplicate clusters : {df.dup_cluster.nunique()}")
    print(f"  images removed          : {n_removed} "
          f"({100 * n_removed / len(df):.1f}%)")

    final = df[df.is_representative].copy().reset_index(drop=True)
    print(f"  final dataset           : {len(final)}")
    print()
    print("  class counts before -> after deduplication:")
    before = df.label.value_counts()
    after = final.label.value_counts()
    comp = pd.DataFrame({"before": before, "after": after})
    comp["removed"] = comp.before - comp.after
    comp["pct"] = (100 * comp.removed / comp.before).round(1)
    print(comp.to_string())

    print()
    print("  groups remaining:", final.group_id.nunique())
    gs = final.groupby("group_id").size()
    print(f"  images per group: median {gs.median():.0f}  max {gs.max()}")

    if not dry_run:
        out = MANIFEST / "dataset.csv"
        final.to_csv(out, index=False)
        print(f"\n  wrote {out}")
        df.to_csv(MANIFEST / "dataset_full_with_dups.csv", index=False)
    else:
        print("\n  --dry-run: nothing written")

    return final


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phash-threshold", type=int, default=6,
                    help="pHash Hamming distance below which two images are "
                         "treated as near-duplicates and grouped")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build(args.phash_threshold, args.dry_run)


if __name__ == "__main__":
    main()