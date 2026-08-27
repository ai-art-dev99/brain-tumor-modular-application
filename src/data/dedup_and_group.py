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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MANIFEST = Path("/workspace/data/manifest")
INSPECT_DIR = MANIFEST / "cluster_montages"

INCLUSION = {
    "glioma":     [("figshare", "glioma")],
    "meningioma": [("figshare", "meningioma"), ("sartaj", "meningioma")],
    "pituitary":  [("figshare", "pituitary"), ("sartaj", "pituitary")],
    "notumor":    [("br35h", "notumor")],
}

EXCLUSION_RATIONALE = {
    ("sartaj", "glioma"):
        "Documented labelling inconsistency in the SARTAJ glioma folder.",
    ("sartaj", "notumor"):
        "500 files reduce to 352 unique hashes; 120 are byte-identical to BR35H.",
    ("br35h", "tumour_unspecified"):
        "BR35H 'yes' images are tumours of unspecified histology.",
    ("br35h", "unknown"):
        "Br35H-Mask-RCNN is a byte-identical repackaging of 'yes'.",
    ("br35h", "unlabelled"):
        "'pred' carries no ground-truth labels.",
}

# Preference order when choosing which member of a duplicate cluster to keep.
SOURCE_PRIORITY = {"figshare": 0, "sartaj": 1, "br35h": 2}

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hex_to_packed(series: pd.Series) -> np.ndarray:
    return np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8)
                     for h in series], dtype=np.uint8)


def dual_hash_edges(ph, dh, threshold, chunk=512):
    """Pairs within `threshold` bits on pHash AND dHash."""
    n = len(ph)
    edges = []
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
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return ra


def load_all() -> pd.DataFrame:
    files = pd.read_csv(MANIFEST / "files_index.csv")
    fig = pd.read_csv(MANIFEST / "figshare_index.csv")

    files = files[["path", "source", "class", "width", "height",
                   "bytes", "phash", "dhash"]].copy()
    files["patient_id"] = ""
    files["mask_path"] = ""

    fig_rows = pd.DataFrame({
        "path": fig.render_path, "source": "figshare", "class": fig["class"],
        "width": fig.width, "height": fig.height, "bytes": 0,
        "phash": fig.phash, "dhash": fig.dhash,
        "patient_id": fig.patient_id, "mask_path": fig.mask_path,
    })
    return pd.concat([files, fig_rows], ignore_index=True)


def build(dedup_t, group_t, dry_run, inspect):
    df_all = load_all()

    # -- 1. curation ----------------------------------------------------------
    print("=" * 70); print("1. Curating"); print("=" * 70)
    keep = np.zeros(len(df_all), bool)
    assigned = np.full(len(df_all), "", dtype=object)
    for cls, pairs in INCLUSION.items():
        for src, src_cls in pairs:
            m = ((df_all.source == src) & (df_all["class"] == src_cls)).to_numpy()
            keep |= m
            assigned[m] = cls
    for (src, src_cls), why in EXCLUSION_RATIONALE.items():
        n = int(((df_all.source == src) & (df_all["class"] == src_cls)).sum())
        if n:
            print(f"  excluded {src}/{src_cls} ({n}): {why}")

    df = df_all[keep].copy().reset_index(drop=True)
    df["label"] = assigned[keep]
    df["patient_id"] = df.patient_id.fillna("")
    df["mask_path"] = df.mask_path.fillna("")
    print(f"\n  curated: {len(df)}")
    print(pd.crosstab(df.source, df.label).to_string())

    ph, dh = hex_to_packed(df.phash), hex_to_packed(df.dhash)

    # -- 2. duplicate clusters ------------------------------------------------
    print(); print("=" * 70)
    print(f"2. Near-duplicate clusters (dual-hash <= {dedup_t})")
    print("=" * 70)

    dup_edges = dual_hash_edges(ph, dh, dedup_t)
    dup = DSU(len(df))
    for i, j in dup_edges:
        dup.union(i, j)
    df["dup_cluster"] = [dup.find(i) for i in range(len(df))]
    print(f"  edges {len(dup_edges)}, clusters {df.dup_cluster.nunique()}")

    # -- 2a. propagate identifiers across each cluster ------------------------
    # Members of a duplicate cluster are the same image; whatever metadata any
    # one of them carries applies to all. Doing this BEFORE representative
    # selection is what stops patient IDs and masks from being thrown away.
    pid_map, mask_map, conflicts = {}, {}, 0
    for cid, g in df.groupby("dup_cluster"):
        pids = set(g.patient_id) - {""}
        if len(pids) > 1:
            conflicts += 1
        if pids:
            pid_map[cid] = sorted(pids)[0]
            m = set(g.mask_path) - {""}
            if m:
                mask_map[cid] = sorted(m)[0]
    df["patient_id"] = [pid_map.get(c, p)
                        for c, p in zip(df.dup_cluster, df.patient_id)]
    df["mask_path"] = [mask_map.get(c, m)
                       for c, m in zip(df.dup_cluster, df.mask_path)]

    gained = (df.patient_id != "").sum()
    print(f"  images with a patient ID after propagation: {gained} "
          f"({100 * gained / len(df):.1f}%)")
    if conflicts:
        print(f"  WARNING: {conflicts} clusters contained two different "
              f"patient IDs;\n           the lexicographically first was kept.")

    # -- 2b. cross-source overlap, a finding in its own right ------------------
    src_combo = (df.groupby("dup_cluster").source
                   .agg(lambda s: " + ".join(sorted(set(s)))))
    multi = src_combo[src_combo.str.contains(r"\+")]
    print(f"\n  clusters spanning >1 repository: {len(multi)}")
    if len(multi):
        print(multi.value_counts().to_string())
        print("  -> the constituent repositories are not independent sources.")

    # -- 2c. representative selection -----------------------------------------
    # Figshare first: it alone carries patient ID and tumour mask, and its
    # render is a lossless PNG of the original 16-bit slice rather than a
    # recompressed JPEG derivative.
    df["_prio"] = df.source.map(SOURCE_PRIORITY).fillna(9)
    df["_hasmask"] = (df.mask_path != "").astype(int)
    order = df.sort_values(["dup_cluster", "_prio", "_hasmask", "bytes"],
                           ascending=[True, True, False, False])
    df["is_representative"] = df.index.isin(
        set(order.drop_duplicates("dup_cluster", keep="first").index))

    n_rm = int((~df.is_representative).sum())
    print(f"\n  removed {n_rm} ({100 * n_rm / len(df):.1f}%)")
    comp = pd.DataFrame({"before": df.label.value_counts(),
                         "after": df[df.is_representative].label.value_counts()})
    comp["removed"] = comp.before - comp.after
    comp["pct"] = (100 * comp.removed / comp.before).round(1)
    print(comp.to_string())
    print("\n  (glioma is the negative control: pure Figshare, distinct slices."
          "\n   A low removal rate there means the threshold is not collapsing"
          "\n   genuinely different scans.)")

    if inspect:
        write_montages(df, inspect)

    sel = df.is_representative.to_numpy()
    final = df[sel].copy().reset_index(drop=True)
    ph_f, dh_f = ph[sel], dh[sel]

    # -- 3. grouping ----------------------------------------------------------
    print(); print("=" * 70)
    print(f"3. Grouping (patient ID, then dual-hash <= {group_t})")
    print("=" * 70)

    grp = DSU(len(final))
    comp_pids: dict[int, set[str]] = defaultdict(set)
    for i, p in enumerate(final.patient_id):
        if p:
            comp_pids[i].add(p)

    def merge(a: int, b: int) -> bool:
        ra, rb = grp.find(a), grp.find(b)
        if ra == rb:
            return False
        pa, pb = comp_pids[ra], comp_pids[rb]
        # Metadata outranks a hash: never fuse two components that are known to
        # belong to different patients.
        if pa and pb and not (pa & pb):
            return False
        new = grp.union(ra, rb)
        merged = pa | pb
        comp_pids[new] = merged
        for r in (ra, rb):
            if r != new:
                comp_pids.pop(r, None)
        return True

    n_pid = 0
    for _, idx in final[final.patient_id != ""].groupby("patient_id").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            n_pid += merge(idx[0], j)
    print(f"  edges from patient ID : {n_pid}")

    sim = dual_hash_edges(ph_f, dh_f, group_t)
    applied = sum(merge(i, j) for i, j in sim)
    print(f"  similarity edges      : {len(sim)} proposed, {applied} applied, "
          f"{len(sim) - applied} blocked by patient-ID conflict")

    final["group_id"] = [f"g{grp.find(i):05d}" for i in range(len(final))]
    sizes = final.groupby("group_id").size()
    print(f"\n  groups {final.group_id.nunique()}; per group: "
          f"median {sizes.median():.0f}  mean {sizes.mean():.1f}  max {sizes.max()}")
    print("\n  groups per class (the real limit on how finely we can split):")
    print(final.groupby("label").group_id.nunique().to_string())

    span = final.groupby("group_id").label.nunique()
    n_span = int((span > 1).sum())
    print(f"\n  groups spanning >1 class: {n_span}")
    if n_span:
        final[final.group_id.isin(span[span > 1].index)].to_csv(
            MANIFEST / "cross_class_groups.csv", index=False)

    n_mask = int((final.mask_path != "").sum())
    print(f"  images with a tumour mask: {n_mask} "
          f"({100 * n_mask / len(final):.1f}%) -- available for the "
          f"interpretability analysis")

    if not dry_run:
        final.drop(columns=["_prio", "_hasmask"]).to_csv(
            MANIFEST / "dataset.csv", index=False)
        df.drop(columns=["_prio", "_hasmask"]).to_csv(
            MANIFEST / "dataset_with_duplicates.csv", index=False)
        print(f"\n  wrote {MANIFEST / 'dataset.csv'} ({len(final)} images)")
    else:
        print("\n  --dry-run: nothing written")
    return final


def write_montages(df, n_clusters, per_row=6, thumb=140):
    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    sizes = df.groupby("dup_cluster").size().sort_values(ascending=False)
    targets = sizes[sizes > 1].head(n_clusters)
    for rank, (cid, n) in enumerate(targets.items(), 1):
        sub = df[df.dup_cluster == cid].head(per_row * 3)
        paths, srcs = sub.path.tolist(), sub.source.tolist()
        cols = min(per_row, len(paths))
        rows = (len(paths) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * thumb), "black")
        for k, p in enumerate(paths):
            try:
                sheet.paste(Image.open(p).convert("RGB").resize((thumb, thumb)),
                            ((k % cols) * thumb, (k // cols) * thumb))
            except Exception:
                pass
        tag = "-".join(sorted(set(srcs)))
        sheet.save(INSPECT_DIR / f"cluster{rank:02d}_n{n}_{tag}.png")
    print(f"  wrote {len(targets)} montages to {INSPECT_DIR}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dedup-threshold", type=int, default=2)
    ap.add_argument("--group-threshold", type=int, default=4)
    ap.add_argument("--inspect", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.group_threshold < a.dedup_threshold:
        raise SystemExit("--group-threshold must be >= --dedup-threshold")
    build(a.dedup_threshold, a.group_threshold, a.dry_run, a.inspect)


if __name__ == "__main__":
    main()