#!/usr/bin/env python3
"""
split.py -- generate nested, group-aware cross-validation splits.

WHY NESTED, AND WHY GROUPED
---------------------------
Grouped: every image belonging to one patient (or, where no identifier exists,
one pseudo-patient cluster) must fall entirely on one side of any partition.
The composite benchmark averages several slices per patient, so an
image-level split places near-identical views of the same brain in both train
and test. That is the single largest source of the optimistic accuracies
reported on this dataset.

Nested: hyperparameters must be chosen without ever consulting the data that
produces the reported number. The outer loop estimates performance; the inner
loop tunes. Selecting a checkpoint or a value of C on the test fold and then
reporting accuracy on that same fold is model selection on the test set.

CONFIGURATIONS
--------------
  main          4-class, the full curated dataset. Groups mix true patient IDs
                (Figshare-derived images) with pseudo-patient clusters
                (BR35H). Comparable in scope to published work.

  figshare_only 3-class, restricted to images carrying a genuine patient ID.
                No 'no tumour' class exists in Figshare. This is the reference
                condition: the only partition where grouping rests entirely on
                metadata rather than inference.

  source_probe  Not a split. Quantifies how many SARTAJ images survive as
                non-duplicates of Figshare, i.e. whether an independent
                source-based test set is constructible at all.

OUTPUT
------
  splits_<config>_outer.csv   path, label, group_id, outer_fold
  splits_<config>_inner.csv   outer_fold, path, inner_fold
  split_report_<config>.txt

Usage
-----
    python split.py --config main --outer 5 --inner 3 --seed 42
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
from sklearn.model_selection import StratifiedGroupKFold

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
        # Restrict to images with a metadata-backed identifier. Note this is
        # not the same as source == 'figshare': duplicate-cluster propagation
        # gave some SARTAJ images a genuine patient ID, and those are
        # legitimately included here.
        sub = df[df.patient_id != ""].copy()
        sub = sub[sub.label != "notumor"]
        return sub.reset_index(drop=True)
    sys.exit(f"unknown config: {config}")


# =============================================================================

def make_splits(df: pd.DataFrame, n_outer: int, n_inner: int,
                seed: int, config: str) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)

    y = df.label.to_numpy()
    g = df.group_id.to_numpy()

    n_groups_min = df.groupby("label").group_id.nunique().min()
    if n_groups_min < n_outer:
        sys.exit(f"ERROR: the smallest class has only {n_groups_min} groups, "
                 f"fewer than --outer {n_outer}. Reduce the number of folds.")

    print(f"==> {config}: {len(df)} images, {df.group_id.nunique()} groups, "
          f"{df.label.nunique()} classes")
    print(f"    smallest class has {n_groups_min} groups")

    # -- outer loop -----------------------------------------------------------
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    df["outer_fold"] = -1
    for k, (_, test_idx) in enumerate(outer.split(df, y, g)):
        df.loc[df.index[test_idx], "outer_fold"] = k

    assert (df.outer_fold >= 0).all(), "some rows were never assigned a fold"

    # -- inner loops ----------------------------------------------------------
    inner_rows = []
    for k in range(n_outer):
        tr = df[df.outer_fold != k].reset_index(drop=True)
        smallest = tr.groupby("label").group_id.nunique().min()
        if smallest < n_inner:
            sys.exit(f"ERROR: outer fold {k} training portion has a class with "
                     f"only {smallest} groups, fewer than --inner {n_inner}.")
        # A different seed per outer fold, derived deterministically, so the
        # inner partitions are not identical copies across outer folds.
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                     random_state=seed + 1000 + k)
        for m, (_, val_idx) in enumerate(inner.split(
                tr, tr.label.to_numpy(), tr.group_id.to_numpy())):
            for p in tr.path.to_numpy()[val_idx]:
                inner_rows.append({"outer_fold": k, "path": p, "inner_fold": m})

    inner_df = pd.DataFrame(inner_rows)

    # -- verification ---------------------------------------------------------
    report = verify(df, inner_df, n_outer, config)

    out_o = SPLITS / f"splits_{config}_outer.csv"
    out_i = SPLITS / f"splits_{config}_inner.csv"
    df[["path", "label", "group_id", "patient_id", "source",
        "mask_path", "outer_fold"]].to_csv(out_o, index=False)
    inner_df.to_csv(out_i, index=False)
    (SPLITS / f"split_report_{config}.txt").write_text(report)

    print(f"\n    wrote {out_o}")
    print(f"    wrote {out_i}")


def verify(df: pd.DataFrame, inner_df: pd.DataFrame,
           n_outer: int, config: str) -> str:
    """
    Assert the properties the whole exercise depends on. A split that silently
    violates them reproduces exactly the flaw being corrected, so these are
    hard failures rather than warnings.
    """
    lines: list[str] = [f"Split verification: {config}", "=" * 60, ""]

    # 1. no group spans two outer folds
    spread = df.groupby("group_id").outer_fold.nunique()
    bad = spread[spread > 1]
    if len(bad):
        sys.exit(f"FATAL: {len(bad)} groups span more than one outer fold.")
    lines.append(f"[ok] no group spans two outer folds "
                 f"({df.group_id.nunique()} groups)")

    # 2. no real patient spans two outer folds
    withpid = df[df.patient_id != ""]
    if len(withpid):
        pspread = withpid.groupby("patient_id").outer_fold.nunique()
        pbad = pspread[pspread > 1]
        if len(pbad):
            sys.exit(f"FATAL: {len(pbad)} patients span more than one fold.")
        lines.append(f"[ok] no patient spans two outer folds "
                     f"({withpid.patient_id.nunique()} patients)")

    # 3. inner validation folds never touch the outer test fold
    for k in range(n_outer):
        test_paths = set(df[df.outer_fold == k].path)
        inner_paths = set(inner_df[inner_df.outer_fold == k].path)
        overlap = test_paths & inner_paths
        if overlap:
            sys.exit(f"FATAL: outer fold {k} test data appears in its own "
                     f"inner loop ({len(overlap)} images).")
    lines.append("[ok] inner loops never see their outer test fold")

    # 4. every class present in every fold
    ct = pd.crosstab(df.outer_fold, df.label)
    if (ct == 0).any().any():
        lines.append("[WARN] a class is absent from at least one fold")
    else:
        lines.append("[ok] every class appears in every fold")

    lines += ["", "Images per outer fold and class:", ct.to_string(), ""]

    gct = df.groupby(["outer_fold", "label"]).group_id.nunique().unstack()
    lines += ["Independent groups per outer fold and class:", gct.to_string(), ""]

    # The ratio of images to groups is the quantity that an image-level split
    # would have silently inflated: it is the average number of correlated
    # samples per independent unit.
    ratio = (ct / gct).round(1)
    lines += ["Images per group (correlated samples per independent unit):",
              ratio.to_string(), ""]

    src = pd.crosstab(df.outer_fold, df.source)
    lines += ["Source composition per fold:", src.to_string(), ""]

    n_mask = df[df.mask_path != ""].groupby("outer_fold").size()
    lines += ["Images with a tumour mask per fold:", n_mask.to_string(), ""]

    text = "\n".join(lines)
    print()
    print(text)
    return text


# =============================================================================

def source_probe(df: pd.DataFrame) -> None:
    """
    How much genuinely independent SARTAJ data survives?

    Representative selection prefers Figshare, so a surviving SARTAJ row is by
    construction an image that was NOT a near-duplicate of any Figshare slice.
    Counting them measures whether a source-based held-out test set can be
    built at all.
    """
    print("=" * 70)
    print("Source-independence probe")
    print("=" * 70)

    surv = df[df.source == "sartaj"]
    print(f"\n  SARTAJ images surviving deduplication: {len(surv)}")
    if len(surv):
        print(surv.label.value_counts().to_string())
        print(f"  independent groups: {surv.group_id.nunique()}")
        # Some survivors inherited a Figshare patient ID through cluster
        # propagation; those are not independent of the training data either.
        indep = surv[surv.patient_id == ""]
        print(f"\n  of which carry NO Figshare patient link: {len(indep)}")
        if len(indep):
            print(indep.label.value_counts().to_string())

    print()
    print("  Assessment:")
    n = len(surv)
    classes = surv.label.nunique() if n else 0
    if n < 300 or classes < 3:
        print("    A source-based held-out test set is NOT constructible from")
        print("    these repositories. SARTAJ is largely a redistribution of the")
        print("    Figshare collection, and BR35H contributes only the")
        print("    'no tumour' class. Genuine external validation requires a")
        print("    dataset outside this benchmark. Report this as a limitation")
        print("    rather than presenting an internal split as external.")
    else:
        print(f"    {n} images across {classes} classes remain. A source-based")
        print("    test set is feasible, though it will be class-incomplete.")


def verify_only(config: str) -> None:
    o = SPLITS / f"splits_{config}_outer.csv"
    i = SPLITS / f"splits_{config}_inner.csv"
    if not o.exists():
        sys.exit(f"{o} not found")
    df = pd.read_csv(o)
    df["patient_id"] = df.patient_id.fillna("")
    df["mask_path"] = df.mask_path.fillna("")
    inner = pd.read_csv(i) if i.exists() else pd.DataFrame(
        columns=["outer_fold", "path", "inner_fold"])
    verify(df, inner, df.outer_fold.nunique(), config)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=["main", "figshare_only", "source_probe"],
                    default="main")
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", metavar="CONFIG",
                    help="re-verify an existing split instead of generating one")
    a = ap.parse_args()

    if a.verify:
        verify_only(a.verify)
        return

    df = load_dataset()
    if a.config == "source_probe":
        source_probe(df)
        return

    make_splits(select(df, a.config), a.outer, a.inner, a.seed, a.config)


if __name__ == "__main__":
    main()