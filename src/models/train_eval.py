#!/usr/bin/env python3
"""
train_eval.py -- nested, group-aware evaluation of the hybrid classifiers.

WHAT THIS REPLACES
------------------
The original pipeline (a) selected the best CNN checkpoint on the test set,
(b) evaluated the CNN on the held-out folder but evaluated the hybrid
classifiers on a random 20% of the *training* folder, so the headline
"96% vs 95%" compared numbers computed on different data, (c) reported
tp/(tp+fp+fn) -- the Jaccard index -- under the name "accuracy", and (d) fixed
SVC(kernel='linear', C=0.025) with no search at all.

Here: hyperparameters are chosen in an inner loop that never sees the outer
test fold; every model is evaluated on the same pooled out-of-fold
predictions; and metrics are named for what they compute.

THE LEAKAGE COMPARISON
----------------------
--split-mode image reruns the identical procedure with a random image-level
split, ignoring patient groups. The gap between the two is a direct estimate
of how much of the published performance on this benchmark comes from
near-duplicate slices of the same patient appearing on both sides of the
split. Report both numbers side by side.

CONFIDENCE INTERVALS ARE RESAMPLED OVER GROUPS, NOT IMAGES
----------------------------------------------------------
With 15.8 correlated images per glioma patient, resampling images treats
15.8 views of one brain as 15.8 independent observations and produces
intervals that are far too narrow. The original manuscript's normal-
approximation intervals (e.g. 0.9529-0.9631) have exactly this problem.
Bootstrapping over groups respects the correlation structure.

Usage
-----
    python train_eval.py --config main --models svm knn rf mlp
    python train_eval.py --config main --split-mode image --tag naive
    python train_eval.py --compare runs/main_grouped runs/main_naive
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar

FEATURES = Path("/workspace/data/features")
SPLITS = Path("/workspace/data/manifest/splits")
RUNS = Path("/workspace/outputs/runs")


# =============================================================================
# Model zoo
# =============================================================================
# class_weight='balanced' rather than oversampling: the classes are only mildly
# imbalanced at image level, and duplicating images inside a group would
# further inflate the effective weight of the few patients that contribute many
# slices.

def model_grid(name: str, seed: int):
    if name == "svm":
        return (SVC(probability=True, class_weight="balanced", random_state=seed),
                {"clf__kernel": ["linear", "rbf"],
                 "clf__C": [0.1, 1, 10, 100],
                 "clf__gamma": ["scale"]})
    if name == "knn":
        return (KNeighborsClassifier(),
                {"clf__n_neighbors": [1, 3, 5, 11, 21],
                 "clf__weights": ["uniform", "distance"]})
    if name == "rf":
        return (RandomForestClassifier(class_weight="balanced_subsample",
                                       random_state=seed, n_jobs=-1),
                {"clf__n_estimators": [300, 600],
                 "clf__max_depth": [None, 10, 20],
                 "clf__min_samples_leaf": [1, 3]})
    if name == "mlp":
        return (MLPClassifier(max_iter=600, early_stopping=True,
                              random_state=seed),
                {"clf__hidden_layer_sizes": [(256,), (512,), (256, 128)],
                 "clf__alpha": [1e-4, 1e-3, 1e-2]})
    if name == "logreg":
        return (LogisticRegression(max_iter=3000, class_weight="balanced",
                                   random_state=seed),
                {"clf__C": [0.01, 0.1, 1, 10]})
    raise SystemExit(f"unknown model: {name}")


# =============================================================================
# Metrics
# =============================================================================

def full_metrics(y_true, y_pred, y_prob, labels) -> dict:
    """Every quantity reviewer point 7 asks for, computed once."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - (tp + fp + fn)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "labels": list(labels),
        "per_class": {},
    }
    for i, lab in enumerate(labels):
        out["per_class"][lab] = {
            "precision": float(prec[i]),
            "recall_sensitivity": float(rec[i]),
            # Specificity is per-class one-vs-rest. In a four-class problem it
            # is bounded near 1 by construction and carries little information;
            # it is reported because it is asked for, not because it
            # discriminates between models.
            "specificity": float(tn[i] / (tn[i] + fp[i])) if (tn[i] + fp[i]) else 0.0,
            "f1": float(f1[i]),
            "support": int(sup[i]),
            "tp": int(tp[i]), "fp": int(fp[i]),
            "fn": int(fn[i]), "tn": int(tn[i]),
            # The metric the original manuscript reported under the name
            # "accuracy". Kept so the old numbers can be reproduced and the
            # discrepancy explained.
            "jaccard": float(tp[i] / (tp[i] + fp[i] + fn[i]))
                       if (tp[i] + fp[i] + fn[i]) else 0.0,
        }

    if y_prob is not None:
        try:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=labels))
            out["roc_auc_ovr_weighted"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted",
                labels=labels))
        except Exception as e:
            out["roc_auc_error"] = str(e)
    return out


def group_bootstrap(y_true, y_pred, groups, fn, n=2000, seed=0, alpha=0.05):
    """Percentile CI, resampling whole groups so within-patient correlation is
    preserved."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in take])
        try:
            vals.append(fn(y_true[sel], y_pred[sel]))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_group_bootstrap(y_true, pred_a, pred_b, groups, fn,
                           n=2000, seed=0, alpha=0.05):
    """CI for the difference between two models on the same resampled groups."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in take])
        try:
            diffs.append(fn(y_true[sel], pred_a[sel]) - fn(y_true[sel], pred_b[sel]))
        except Exception:
            continue
    d = np.array(diffs)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p-value: how often the difference crosses zero.
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


# =============================================================================
# Splits
# =============================================================================

def load_data(config: str, split_mode: str, seed: int, n_folds: int):
    X = np.load(FEATURES / f"{config}_frozen.npy")
    df = pd.read_csv(FEATURES / f"{config}_index.csv")

    if split_mode == "grouped":
        inner = pd.read_csv(SPLITS / f"splits_{config}_inner.csv")
        return X, df, inner

    # Image-level split: deliberately ignores groups. This is the flawed
    # procedure being quantified, not an alternative worth using.
    print("  !! image-level split: patient groups are IGNORED by design")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df = df.copy()
    df["outer_fold"] = -1
    for k, (_, te) in enumerate(skf.split(df, df.label)):
        df.loc[df.index[te], "outer_fold"] = k

    rows = []
    for k in range(n_folds):
        tr = df[df.outer_fold != k]
        s2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + k)
        for m, (_, va) in enumerate(s2.split(tr, tr.label)):
            for p in tr.path.to_numpy()[va]:
                rows.append({"outer_fold": k, "path": p, "inner_fold": m})
    return X, df, pd.DataFrame(rows)


def inner_cv_for_fold(train_df: pd.DataFrame, inner: pd.DataFrame, k: int):
    """Turn the precomputed inner assignment into (train_idx, val_idx) pairs."""
    sub = inner[inner.outer_fold == k]
    fold_of = dict(zip(sub.path, sub.inner_fold))
    assign = train_df.path.map(fold_of).to_numpy()
    if pd.isna(assign).any():
        raise SystemExit("some training rows have no inner-fold assignment")
    pairs = []
    for m in sorted(set(assign)):
        va = np.where(assign == m)[0]
        tr = np.where(assign != m)[0]
        pairs.append((tr, va))
    return pairs


# =============================================================================

def run(config, models, split_mode, seed, n_folds, tag, n_boot):
    X, df, inner = load_data(config, split_mode, seed, n_folds)
    labels = sorted(df.label.unique())
    y = df.label.to_numpy()
    groups = df.group_id.to_numpy()
    folds = df.outer_fold.to_numpy()

    run_id = tag or f"{config}_{split_mode}"
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> run_id={run_id}  {len(df)} images, "
          f"{df.group_id.nunique()} groups, {len(labels)} classes")

    all_metrics, preds_table = {}, {"path": df.path, "y_true": y,
                                    "group_id": groups, "outer_fold": folds}

    for name in models:
        print(f"\n--- {name}")
        est, grid = model_grid(name, seed)
        pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])

        oof_pred = np.empty(len(df), dtype=object)
        oof_prob = np.zeros((len(df), len(labels)))
        chosen, t0 = [], time.time()

        for k in range(n_folds):
            tr_mask, te_mask = folds != k, folds == k
            tr_df = df[tr_mask].reset_index(drop=True)
            cv = inner_cv_for_fold(tr_df, inner, k)

            gs = GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy",
                              n_jobs=-1, refit=True)
            gs.fit(X[tr_mask], y[tr_mask])
            chosen.append(gs.best_params_)

            oof_pred[te_mask] = gs.predict(X[te_mask])
            prob = gs.predict_proba(X[te_mask])
            # Align probability columns to the global label order.
            col = [list(gs.best_estimator_.classes_).index(l) for l in labels]
            oof_prob[te_mask] = prob[:, col]
            print(f"    fold {k}: inner best {gs.best_params_}  "
                  f"outer acc {accuracy_score(y[te_mask], oof_pred[te_mask]):.4f}")

        elapsed = time.time() - t0
        oof_pred = oof_pred.astype(str)

        m = full_metrics(y, oof_pred, oof_prob, labels)
        m["fit_predict_seconds"] = round(elapsed, 1)
        m["chosen_hyperparameters_per_fold"] = [
            {k2.replace("clf__", ""): v for k2, v in c.items()} for c in chosen]

        for metric_name, fn in [
            ("accuracy", accuracy_score),
            ("balanced_accuracy", balanced_accuracy_score),
            ("macro_f1", lambda a, b: f1_score(a, b, average="macro",
                                               zero_division=0)),
        ]:
            lo, hi = group_bootstrap(y, oof_pred, groups, fn, n_boot, seed)
            m[f"{metric_name}_ci95"] = [lo, hi]

        all_metrics[name] = m
        preds_table[f"pred_{name}"] = oof_pred
        for i, lab in enumerate(labels):
            preds_table[f"prob_{name}_{lab}"] = oof_prob[:, i]

        print(f"    accuracy {m['accuracy']:.4f} "
              f"[{m['accuracy_ci95'][0]:.4f}, {m['accuracy_ci95'][1]:.4f}]  "
              f"balanced {m['balanced_accuracy']:.4f}  "
              f"macroF1 {m['macro_f1']:.4f}  {elapsed:.0f}s")

    # -- pairwise significance -------------------------------------------------
    print("\n--- pairwise comparisons")
    comparisons = []
    for a, b in combinations(models, 2):
        pa, pb = preds_table[f"pred_{a}"], preds_table[f"pred_{b}"]
        ca, cb = (pa == y), (pb == y)
        n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
        res = mcnemar(np.array([[int((ca & cb).sum()), n10],
                                [n01, int((~ca & ~cb).sum())]]), exact=True)
        d, lo, hi, pb_ = paired_group_bootstrap(
            y, pa, pb, groups, accuracy_score, n_boot, seed)
        comparisons.append({
            "model_a": a, "model_b": b,
            "a_only_correct": n10, "b_only_correct": n01,
            "mcnemar_exact_p": float(res.pvalue),
            "acc_diff": d, "acc_diff_ci95": [lo, hi], "bootstrap_p": pb_,
        })
        print(f"    {a} vs {b}: diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"McNemar p={res.pvalue:.4f}  bootstrap p={pb_:.4f}")

    # Holm-Bonferroni across the family of pairwise tests.
    order = np.argsort([c["mcnemar_exact_p"] for c in comparisons])
    n_cmp = len(comparisons)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(running, comparisons[i]["mcnemar_exact_p"] * (n_cmp - rank)))
        comparisons[i]["mcnemar_p_holm"] = float(adj)
        running = adj

    print("\n  NOTE: McNemar assumes independent observations. With ~4-16")
    print("  correlated slices per patient that assumption is violated, so the")
    print("  group-level paired bootstrap is the primary test here and McNemar")
    print("  is reported because it was requested. Where they disagree, trust")
    print("  the bootstrap.")

    # -- write -----------------------------------------------------------------
    payload = {
        "run_id": run_id, "config": config, "split_mode": split_mode,
        "seed": seed, "n_folds": n_folds, "n_images": len(df),
        "n_groups": int(df.group_id.nunique()), "labels": labels,
        "class_counts": df.label.value_counts().to_dict(),
        "models": all_metrics, "comparisons": comparisons,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(preds_table).to_csv(out_dir / "predictions.csv", index=False)

    summary = pd.DataFrame([{
        "model": k,
        "accuracy": v["accuracy"],
        "acc_ci_lo": v["accuracy_ci95"][0], "acc_ci_hi": v["accuracy_ci95"][1],
        "balanced_accuracy": v["balanced_accuracy"],
        "macro_f1": v["macro_f1"],
        "roc_auc_macro": v.get("roc_auc_ovr_macro", float("nan")),
        "seconds": v["fit_predict_seconds"],
    } for k, v in all_metrics.items()]).sort_values("accuracy", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"\n{summary.to_string(index=False)}")
    print(f"\n  wrote {out_dir}/")

    for name, v in all_metrics.items():
        print(f"\n  per-class -- {name}")
        print(pd.DataFrame(v["per_class"]).T[
            ["support", "precision", "recall_sensitivity",
             "specificity", "f1"]].round(3).to_string())


def compare_runs(dirs: list[str]) -> None:
    """Side-by-side table across runs, e.g. grouped vs image-level split."""
    rows = []
    for d in dirs:
        p = Path(d)
        if not (p / "metrics.json").exists():
            p = RUNS / d
        meta = json.loads((p / "metrics.json").read_text())
        for name, m in meta["models"].items():
            rows.append({
                "run": meta["run_id"], "split": meta["split_mode"],
                "model": name, "accuracy": m["accuracy"],
                "ci_lo": m["accuracy_ci95"][0], "ci_hi": m["accuracy_ci95"][1],
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
            })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))

    if t.split.nunique() > 1:
        piv = t.pivot_table(index="model", columns="split", values="accuracy")
        if {"grouped", "image"} <= set(piv.columns):
            piv["leakage_inflation"] = (piv["image"] - piv["grouped"]).round(4)
            print("\nAccuracy inflation attributable to image-level splitting:")
            print(piv.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="main")
    ap.add_argument("--models", nargs="+",
                    default=["svm", "knn", "rf", "mlp", "logreg"])
    ap.add_argument("--split-mode", choices=["grouped", "image"],
                    default="grouped")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--compare", nargs="+", default=None)
    a = ap.parse_args()

    if a.compare:
        compare_runs(a.compare)
    else:
        run(a.config, a.models, a.split_mode, a.seed, a.folds, a.tag, a.bootstrap)


if __name__ == "__main__":
    main()