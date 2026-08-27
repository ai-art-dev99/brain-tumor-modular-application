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
from PIL import Image

MANIFEST = Path("/workspace/data/manifest")
INSPECT_DIR = MANIFEST / "cluster_montages"

# =============================================================================
# Inclusion / exclusion rules (reviewer point 2 asks for these explicitly)
# =============================================================================

INCLUSION = {
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
        "120 of which are byte-identical to BR35H images.",
    ("br35h", "tumour_unspecified"):
        "BR35H 'yes' images are tumours of unspecified histology.",
    ("br35h", "unknown"):
        "Br35H-Mask-RCNN is a byte-identical repackaging of 'yes'.",
    ("br35h", "unlabelled"):
        "'pred' carries no ground-truth labels.",
}


# =============================================================================
# Hashing utilities
# =============================================================================

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hex_to_packed(series: pd.Series) -> np.ndarray:
    """
    Hex hash strings -> (n, 8) uint8.

    Bit ordering does not matter here: Hamming distance is invariant under any
    permutation applied identically to both operands, and hashes are only ever
    compared against hashes of the same family.
    """
    return np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8)
                     for h in series], dtype=np.uint8)


def dual_hash_edges(ph: np.ndarray, dh: np.ndarray, threshold: int,
                    chunk: int = 512) -> list[tuple[int, int]]:
    """Index pairs within `threshold` bits on pHash AND on dHash."""
    n = len(ph)
    edges: list[tuple[int, int]] = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dp = POPCOUNT[ph[s:e, None, :] ^ ph[None, :, :]].sum(axis=2)
        dd = POPCOUNT[dh[s:e, None, :] ^ dh[None, :, :]].sum(axis=2)
        rows, cols = np.where((dp <= threshold) & (dd <= threshold))
        for r, c in zip(rows, cols):
            i, j = s + int(r), int(c)
            if i < j:
                edges.append((i, j))
    return edges


class DSU:
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


# =============================================================================

def load_all() -> pd.DataFrame:
    """Unify the Figshare index and the loose-file index into one table."""
    files = pd.read_csv(MANIFEST / "files_index.csv")
    fig = pd.read_csv(MANIFEST / "figshare_index.csv")

    files = files[["path", "source", "class", "width", "height",
                   "bytes", "sha256", "phash", "dhash"]].copy()
    files["patient_id"] = ""
    files["mask_path"] = ""

    fig_rows = pd.DataFrame({
        "path": fig.render_path,
        "source": "figshare",
        "class": fig["class"],
        "width": fig.width,
        "height": fig.height,
        # Renders are lossless PNG of identical dimensions, so file size
        # carries no quality signal here; a constant keeps representative
        # selection from preferring Figshare arbitrarily.
        "bytes": 0,
        "sha256": "",
        "phash": fig.phash,
        "dhash": fig.dhash,
        "patient_id": fig.patient_id,
        "mask_path": fig.mask_path,
    })

    df = pd.concat([files, fig_rows], ignore_index=True)
    return df


def build(dedup_t: int, group_t: int, dry_run: bool, inspect: int) -> pd.DataFrame:
    df_all = load_all()

    # -- 1. curation ----------------------------------------------------------
    print("=" * 70)
    print("1. Curating the dataset")
    print("=" * 70)

    keep = np.zeros(len(df_all), dtype=bool)
    assigned = np.full(len(df_all), "", dtype=object)
    for cls, pairs in INCLUSION.items():
        for src, src_cls in pairs:
            m = ((df_all.source == src) & (df_all["class"] == src_cls)).to_numpy()
            if not m.any():
                print(f"  WARNING: no images matched {src}/{src_cls}")
            keep |= m
            assigned[m] = cls
            print(f"  {cls:<12} <- {src:<10} {src_cls:<14} {int(m.sum()):5d}")

    print("\n  excluded:")
    for (src, src_cls), why in EXCLUSION_RATIONALE.items():
        n = int(((df_all.source == src) & (df_all["class"] == src_cls)).sum())
        if n:
            print(f"    {src}/{src_cls} ({n}): {why}")

    df = df_all[keep].copy().reset_index(drop=True)
    df["label"] = assigned[keep]
    print(f"\n  curated total: {len(df)}")
    print(df.label.value_counts().to_string())
    print("\n  by source:")
    print(pd.crosstab(df.source, df.label).to_string())

    ph = hex_to_packed(df.phash)
    dh = hex_to_packed(df.dhash)

    has_pid = df.patient_id.fillna("") != ""
    print(f"\n  with true patient ID: {int(has_pid.sum())} "
          f"({100 * has_pid.mean():.1f}%), "
          f"{df.patient_id[has_pid].nunique()} patients")

    # -- 2. deduplication (tight) ---------------------------------------------
    print()
    print("=" * 70)
    print(f"2. Near-duplicate removal (dual-hash threshold = {dedup_t})")
    print("=" * 70)

    dup_edges = dual_hash_edges(ph, dh, dedup_t)
    dup = DSU(len(df))
    for i, j in dup_edges:
        dup.union(i, j)
    df["dup_cluster"] = [dup.find(i) for i in range(len(df))]

    order = df.sort_values(["dup_cluster", "bytes"], ascending=[True, False])
    rep_idx = set(order.drop_duplicates("dup_cluster", keep="first").index)
    df["is_representative"] = df.index.isin(rep_idx)

    n_rm = int((~df.is_representative).sum())
    print(f"  edges           : {len(dup_edges)}")
    print(f"  clusters        : {df.dup_cluster.nunique()}")
    print(f"  images removed  : {n_rm} ({100 * n_rm / len(df):.1f}%)")

    comp = pd.DataFrame({
        "before": df.label.value_counts(),
        "after": df[df.is_representative].label.value_counts(),
    })
    comp["removed"] = comp.before - comp.after
    comp["pct"] = (100 * comp.removed / comp.before).round(1)
    print("\n" + comp.to_string())

    if inspect:
        write_montages(df, dup_edges, inspect)

    final = df[df.is_representative].copy().reset_index(drop=True)
    ph_f = ph[df.is_representative.to_numpy()]
    dh_f = dh[df.is_representative.to_numpy()]

    # -- 3. grouping (loose) --------------------------------------------------
    print()
    print("=" * 70)
    print(f"3. Grouping (patient ID + dual-hash threshold = {group_t})")
    print("=" * 70)

    grp = DSU(len(final))

    n_pid = 0
    fpid = final.patient_id.fillna("")
    for _, idx in final[fpid != ""].groupby("patient_id").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            grp.union(idx[0], j)
            n_pid += 1
    print(f"  edges from patient ID : {n_pid}")

    grp_edges = dual_hash_edges(ph_f, dh_f, group_t)
    for i, j in grp_edges:
        grp.union(i, j)
    print(f"  edges from similarity : {len(grp_edges)}")

    final["group_id"] = [f"g{grp.find(i):05d}" for i in range(len(final))]
    sizes = final.groupby("group_id").size()
    print(f"\n  groups          : {final.group_id.nunique()}")
    print(f"  images per group: min {sizes.min()}  median {sizes.median():.0f}  "
          f"mean {sizes.mean():.1f}  max {sizes.max()}")

    # Independent units per class is the real constraint on how finely the data
    # can be split. Fewer groups than folds in any class makes CV impossible.
    print("\n  groups per class:")
    print(final.groupby("label").group_id.nunique().to_string())

    fpid = final.patient_id.fillna("")
    merged = final[fpid != ""].groupby("group_id").patient_id.nunique()
    multi = merged[merged > 1]
    if len(multi):
        print(f"\n  groups merging >1 patient: {len(multi)} "
              f"(largest merges {multi.max()})")
        if multi.max() > 10:
            print("  -> a group swallowing this many patients suggests "
                  "--group-threshold\n     is too loose; it starves the split "
                  "of independent units.")

    cls_span = final.groupby("group_id").label.nunique()
    n_span = int((cls_span > 1).sum())
    print(f"\n  groups spanning >1 class: {n_span}")
    if n_span:
        bad = final[final.group_id.isin(cls_span[cls_span > 1].index)]
        bad.to_csv(MANIFEST / "cross_class_groups.csv", index=False)
        print(f"  -> {MANIFEST / 'cross_class_groups.csv'} -- inspect these")

    if not dry_run:
        final.to_csv(MANIFEST / "dataset.csv", index=False)
        df.to_csv(MANIFEST / "dataset_with_duplicates.csv", index=False)
        print(f"\n  wrote {MANIFEST / 'dataset.csv'} ({len(final)} images)")
    else:
        print("\n  --dry-run: nothing written")

    return final


def write_montages(df: pd.DataFrame, edges: list[tuple[int, int]],
                   n_clusters: int, per_row: int = 6, thumb: int = 140) -> None:
    """
    Render the largest duplicate clusters so the removal decision can be
    checked by eye rather than taken on trust.
    """
    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    sizes = df.groupby("dup_cluster").size().sort_values(ascending=False)
    targets = sizes[sizes > 1].head(n_clusters)
    print(f"\n  writing {len(targets)} montages to {INSPECT_DIR}")

    for rank, (cid, n) in enumerate(targets.items(), 1):
        paths = df[df.dup_cluster == cid].path.tolist()[:per_row * 3]
        cols = min(per_row, len(paths))
        rows = (len(paths) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * thumb), "black")
        for k, p in enumerate(paths):
            try:
                im = Image.open(p).convert("RGB").resize((thumb, thumb))
            except Exception:
                continue
            sheet.paste(im, ((k % cols) * thumb, (k // cols) * thumb))
        sheet.save(INSPECT_DIR / f"cluster{rank:02d}_n{n}.png")

    print("  -> if images in a montage are visibly different scans, the "
          "threshold is\n     too loose and real data is being discarded.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedup-threshold", type=int, default=2)
    ap.add_argument("--group-threshold", type=int, default=8)
    ap.add_argument("--inspect", type=int, default=0,
                    help="write montages of the N largest duplicate clusters")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.group_threshold < a.dedup_threshold:
        raise SystemExit("--group-threshold must be >= --dedup-threshold")
    build(a.dedup_threshold, a.group_threshold, a.dry_run, a.inspect)


if __name__ == "__main__":
    main()