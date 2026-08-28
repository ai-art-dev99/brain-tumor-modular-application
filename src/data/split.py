#!/usr/bin/env python3
"""
split.py (v2) -- generate nested cross-validation splits, grouped or
deliberately ungrouped.

WHY AN UNGROUPED OPTION EXISTS
------------------------------
--ignore-groups reproduces the flawed procedure used throughout the published
literature on this benchmark: a random split at the level of individual images,
which scatters slices of one patient across train and test. It is generated
here only so that the identical model, features and metrics can be run on both
partitions. The difference between the two is the estimate of how much
published performance on this dataset is attributable to leakage.

When --ignore-groups is set, the verification step does not abort on a group
that spans folds. Instead it MEASURES the contamination -- how many patients
and groups are split across the boundary, and how many test images have a
same-patient sibling in training -- because those counts are themselves a
result to report.

CHANGES FROM v1
---------------
- --ignore-groups and --tag added, so a flawed baseline can be written
  alongside the grouped split without overwriting it.
- Verification is strict for grouped splits (violations abort) and diagnostic
  for ungrouped ones (violations are counted and reported).
- The leakage summary now reports the fraction of test images that have a
  same-patient sibling in the training portion, which is the quantity that
  actually drives the inflation.

Usage
-----
    python split.py --config main --outer 5 --inner 3 --seed 42
    python split.py --config main --ignore-groups --tag main_imagelevel
    python split.py --config figshare_only
    python split.py --config source_probe
    python split.py --verify main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

MANIFEST = Path("/workspace/data/manifest")
SPLITS = MANIFEST / "splits"


def load_dataset() -> pd.DataFrame:
    p = MANIFEST / "dataset.csv"
    if not p.exists():
        sys.exit(f"ERROR: {p} not found. Run dedup_and_group.py without --dry-run.")
    df = pd.read_csv(p)
    df["patient_id"] = df.patient_id.fillna("")
    df["mask_path"] = df.mask_path.fillna("")
    return df


def select(df: pd.DataFrame, config: str) -> pd.DataFrame:
    if config == "main":
        return df.reset_index(drop=True)
    if config == "figshare_only":
        # Images carrying a metadata-backed identifier. Not identical to
        # source == 'figshare': duplicate-cluster propagation gave some SARTAJ
        # images a genuine patient ID, and those belong here too.
        sub = df[(df.patient_id != "") & (df.label != "notumor")]
        return sub.reset_index(drop=True)
    sys.exit(f"unknown config: {config}")


# =============================================================================

def make_splits(df: pd.DataFrame, n_outer: int, n_inner: int, seed: int,
                name: str, ignore_groups: bool) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)

    y = df.label.to_numpy()
    g = df.group_id.to_numpy()

    if ignore_groups:
        print("  !! --ignore-groups: patient structure is discarded ON PURPOSE.")
        print("     This partition is the flawed baseline, not a usable split.")
        outer_iter = StratifiedKFold(
            n_splits=n_outer, shuffle=True, random_state=seed).split(df, y)
    else:
        smallest = df.groupby("label").group_id.nunique().min()
        if smallest < n_outer:
            sys.exit(f"ERROR: smallest class has {smallest} groups < "
                     f"--outer {n_outer}.")
        print(f"    smallest class has {smallest} groups")
        outer_iter = StratifiedGroupKFold(
            n_splits=n_outer, shuffle=True, random_state=seed).split(df, y, g)

    df = df.copy()
    df["outer_fold"] = -1
    for k, (_, te) in enumerate(outer_iter):
        df.loc[df.index[te], "outer_fold"] = k
    assert (df.outer_fold >= 0).all(), "some rows were never assigned a fold"

    # -- inner loops ----------------------------------------------------------
    inner_rows = []
    for k in range(n_outer):
        tr = df[df.outer_fold != k].reset_index(drop=True)
        # A distinct but deterministic seed per outer fold, so inner partitions
        # are not identical copies across the outer loop.
        rs = seed + 1000 + k
        if ignore_groups:
            it = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                 random_state=rs).split(tr, tr.label)
        else:
            s = tr.groupby("label").group_id.nunique().min()
            if s < n_inner:
                sys.exit(f"ERROR: outer fold {k} training portion has a class "
                         f"with {s} groups < --inner {n_inner}.")
            it = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                      random_state=rs).split(
                tr, tr.label, tr.group_id)
        for m, (_, va) in enumerate(it):
            for p in tr.path.to_numpy()[va]:
                inner_rows.append({"outer_fold": k, "path": p, "inner_fold": m})

    inner_df = pd.DataFrame(inner_rows)
    report = verify(df, inner_df, n_outer, name, strict=not ignore_groups)

    out_o = SPLITS / f"splits_{name}_outer.csv"
    df[["path", "label", "group_id", "patient_id", "source",
        "mask_path", "outer_fold"]].to_csv(out_o, index=False)
    inner_df.to_csv(SPLITS / f"splits_{name}_inner.csv", index=False)
    (SPLITS / f"split_report_{name}.txt").write_text(report)
    print(f"\n    wrote {out_o}")
    print(f"    wrote {SPLITS / f'splits_{name}_inner.csv'}")


def verify(df: pd.DataFrame, inner_df: pd.DataFrame, n_outer: int,
           name: str, strict: bool) -> str:
    """
    Strict mode: any violation aborts, because a grouped split that silently
    leaks reproduces the exact flaw being corrected.
    Diagnostic mode: violations are counted and reported, because for the
    ungrouped baseline they are the measurement of interest.
    """
    lines = [f"Split verification: {name}",
             f"mode: {'strict (grouped)' if strict else 'diagnostic (ungrouped)'}",
             "=" * 60, ""]

    # -- group containment ----------------------------------------------------
    spread = df.groupby("group_id").outer_fold.nunique()
    bad_groups = spread[spread > 1]
    if strict:
        if len(bad_groups):
            sys.exit(f"FATAL: {len(bad_groups)} groups span >1 outer fold.")
        lines.append(f"[ok] no group spans two outer folds "
                     f"({df.group_id.nunique()} groups)")
    else:
        pct = 100 * len(bad_groups) / df.group_id.nunique()
        lines.append(f"[leak] {len(bad_groups)} of {df.group_id.nunique()} "
                     f"groups span >1 outer fold ({pct:.1f}%)")

    # -- patient containment --------------------------------------------------
    withpid = df[df.patient_id != ""]
    if len(withpid):
        ps = withpid.groupby("patient_id").outer_fold.nunique()
        bad_p = ps[ps > 1]
        if strict:
            if len(bad_p):
                sys.exit(f"FATAL: {len(bad_p)} patients span >1 outer fold.")
            lines.append(f"[ok] no patient spans two outer folds "
                         f"({withpid.patient_id.nunique()} patients)")
        else:
            pct = 100 * len(bad_p) / withpid.patient_id.nunique()
            lines.append(f"[leak] {len(bad_p)} of "
                         f"{withpid.patient_id.nunique()} patients span >1 "
                         f"outer fold ({pct:.1f}%)")

    # -- the quantity that actually drives inflation --------------------------
    # For each fold, how many test images belong to a group that also appears
    # in that fold's training portion? Those images have a near-identical
    # sibling the model has already memorised.
    contaminated = []
    for k in range(n_outer):
        te = df[df.outer_fold == k]
        tr_groups = set(df[df.outer_fold != k].group_id)
        n_bad = int(te.group_id.isin(tr_groups).sum())
        contaminated.append((k, len(te), n_bad, 100 * n_bad / max(len(te), 1)))
    tot_bad = sum(c[2] for c in contaminated)
    lines += ["", "Test images with a same-group sibling in training:"]
    for k, n, nb, pc in contaminated:
        lines.append(f"  fold {k}: {nb:5d} / {n:5d}  ({pc:5.1f}%)")
    lines.append(f"  overall: {tot_bad} / {len(df)} "
                 f"({100 * tot_bad / len(df):.1f}%)")
    if not strict and tot_bad:
        lines.append("  -> this is the contamination the grouped split removes.")

    # -- inner loops ----------------------------------------------------------
    for k in range(n_outer):
        overlap = (set(df[df.outer_fold == k].path)
                   & set(inner_df[inner_df.outer_fold == k].path))
        if overlap:
            sys.exit(f"FATAL: outer fold {k} test data appears in its own "
                     f"inner loop ({len(overlap)} images). This is a bug, not "
                     f"a property of the split mode.")
    lines.append("")
    lines.append("[ok] inner loops never see their outer test fold")

    ct = pd.crosstab(df.outer_fold, df.label)
    if (ct == 0).any().any():
        lines.append("[WARN] a class is absent from at least one fold")
    else:
        lines.append("[ok] every class appears in every fold")

    lines += ["", "Images per outer fold and class:", ct.to_string(), ""]
    gct = df.groupby(["outer_fold", "label"]).group_id.nunique().unstack()
    lines += ["Independent groups per outer fold and class:", gct.to_string(), ""]
    lines += ["Images per group (correlated samples per independent unit):",
              (ct / gct).round(1).to_string(), ""]
    lines += ["Source composition per fold:",
              pd.crosstab(df.outer_fold, df.source).to_string(), ""]
    if (df.mask_path != "").any():
        lines += ["Images with a tumour mask per fold:",
                  df[df.mask_path != ""].groupby("outer_fold").size().to_string(),
                  ""]

    text = "\n".join(lines)
    print()
    print(text)
    return text


# =============================================================================

def source_probe(df: pd.DataFrame) -> None:
    """How much genuinely independent SARTAJ data survives deduplication?"""
    print("=" * 70)
    print("Source-independence probe")
    print("=" * 70)
    surv = df[df.source == "sartaj"]
    print(f"\n  SARTAJ images surviving deduplication: {len(surv)}")
    if len(surv):
        print(surv.label.value_counts().to_string())
        print(f"  independent groups: {surv.group_id.nunique()}")
        indep = surv[surv.patient_id == ""]
        print(f"\n  with no Figshare patient link: {len(indep)}")
    print()
    if len(surv) < 300 or surv.label.nunique() < 3:
        print("  A source-based held-out test set is NOT constructible here.")
        print("  SARTAJ is largely a redistribution of Figshare, and BR35H")
        print("  contributes only the 'no tumour' class. Genuine external")
        print("  validation requires a dataset outside this benchmark.")


def verify_only(name: str) -> None:
    o = SPLITS / f"splits_{name}_outer.csv"
    if not o.exists():
        sys.exit(f"{o} not found")
    df = pd.read_csv(o)
    df["patient_id"] = df.patient_id.fillna("")
    df["mask_path"] = df.mask_path.fillna("")
    i = SPLITS / f"splits_{name}_inner.csv"
    inner = pd.read_csv(i) if i.exists() else pd.DataFrame(
        columns=["outer_fold", "path", "inner_fold"])
    # Re-verification is always diagnostic: report what is there rather than
    # aborting on a file that was generated deliberately.
    verify(df, inner, df.outer_fold.nunique(), name, strict=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=["main", "figshare_only", "source_probe"],
                    default="main")
    ap.add_argument("--ignore-groups", action="store_true",
                    help="image-level split; generates the flawed baseline")
    ap.add_argument("--tag", default=None,
                    help="output name (defaults to --config)")
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", metavar="NAME")
    a = ap.parse_args()

    if a.verify:
        verify_only(a.verify)
        return

    df = load_dataset()
    if a.config == "source_probe":
        source_probe(df)
        return

    name = a.tag or a.config
    if a.ignore_groups and name == a.config:
        sys.exit("Refusing to overwrite the grouped split. Pass --tag, e.g. "
                 f"--tag {a.config}_imagelevel")

    sub = select(df, a.config)
    print(f"==> {name}: {len(sub)} images, {sub.group_id.nunique()} groups, "
          f"{sub.label.nunique()} classes")
    make_splits(sub, a.outer, a.inner, a.seed, name, a.ignore_groups)


if __name__ == "__main__":
    main()