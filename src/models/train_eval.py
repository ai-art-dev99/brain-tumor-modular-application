#!/usr/bin/env python3
"""
train_eval.py (v2) -- nested, group-aware evaluation.

CHANGES FROM v1
---------------
1. MLPClassifier(early_stopping=True) removed. scikit-learn carves its own
   internal validation split at random, which ignores the patient grouping and
   places sibling slices on both sides of it. The split is inside the training
   fold so nothing reaches the outer test set, but the stopping point was being
   chosen against partially memorised data. max_iter is fixed instead.

2. SVC no longer uses probability=True. That option fits Platt scaling through
   an internal 5-fold CV which is likewise not group-aware -- and those are the
   very probabilities the deferral analysis rests on. Calibration is now done
   explicitly with CalibratedClassifierCV over the same group-disjoint inner
   folds used for tuning.

3. Confidence intervals now cover every reported metric, not just three.
   A single bootstrap pass computes all of them, so the cost is unchanged.

4. Calibration quality is reported (multiclass Brier score, expected
   calibration error), because a deferral claim depends on the scores being
   meaningful and not merely ordered.

5. Per-fold results are exported. Pooled out-of-fold estimates hide
   between-fold spread, which on this dataset is large: with ~47 test patients
   per fold and up to 38 slices each, a handful of difficult patients moves
   accuracy by several points. A single random split can land anywhere in that
   range, which is one route by which single-split studies report 98-99%.

Usage
-----
    python train_eval.py --config main --models svm knn rf mlp logreg
    python train_eval.py --config main --split-mode image --tag main_naive
    python train_eval.py --compare main_grouped main_naive
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
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

# Estimators whose native probabilities come from an internal, non-group-aware
# resampling scheme and must be recalibrated over group-disjoint folds instead.
NEEDS_CALIBRATION = {"svm"}


# =============================================================================
# Model zoo
# =============================================================================

def model_grid(name: str, seed: int):
    if name == "svm":
        # probability=False: calibration is applied afterwards, group-aware.
        return (SVC(probability=False, class_weight="balanced",
                    random_state=seed),
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
        # early_stopping deliberately off: see module docstring.
        return (MLPClassifier(max_iter=800, early_stopping=False,
                              random_state=seed),
                {"clf__hidden_layer_sizes": [(256,), (512,), (256, 128)],
                 "clf__alpha": [1e-4, 1e-3, 1e-2]})
    if name == "logreg":
        return (LogisticRegression(max_iter=3000, class_weight="balanced",
                                   random_state=seed),
                {"clf__C": [0.01, 0.1, 1, 10]})
    raise SystemExit(f"unknown model: {name}")


def fit_tuned(name: str, X, y, cv_pairs, seed: int):
    """
    Tune on group-disjoint inner folds, then calibrate on the same folds where
    the estimator's own probability machinery would not respect grouping.
    Returns (fitted_estimator, best_params).
    """
    est, grid = model_grid(name, seed)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
    gs = GridSearchCV(pipe, grid, cv=cv_pairs, scoring="balanced_accuracy",
                      n_jobs=-1, refit=True)
    gs.fit(X, y)

    if name in NEEDS_CALIBRATION:
        cal = CalibratedClassifierCV(clone(gs.best_estimator_),
                                     method="sigmoid", cv=cv_pairs)
        cal.fit(X, y)
        return cal, gs.best_params_
    return gs.best_estimator_, gs.best_params_


# =============================================================================
# Metrics
# =============================================================================

def multiclass_brier(y_idx: np.ndarray, P: np.ndarray, k: int) -> float:
    """Mean squared error between the probability vector and the one-hot truth."""
    return float(((P - np.eye(k)[y_idx]) ** 2).sum(1).mean())


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15) -> float:
    """Weighted mean gap between confidence and accuracy across bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def metric_bundle(y_true, y_pred, y_prob, labels) -> dict:
    """Every scalar metric, computed once. Used both for point estimates and
    inside the bootstrap loop, so the two can never drift apart."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - (tp + fp + fn)
    spec = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float),
                     where=(tn + fp) > 0)

    out["macro_precision"] = float(prec.mean())
    out["macro_recall"] = float(rec.mean())
    out["macro_specificity"] = float(spec.mean())
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro",
                                     zero_division=0))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted",
                                        zero_division=0))
    for i, lab in enumerate(labels):
        out[f"precision::{lab}"] = float(prec[i])
        out[f"recall::{lab}"] = float(rec[i])
        out[f"specificity::{lab}"] = float(spec[i])
        out[f"f1::{lab}"] = float(f1[i])

    if y_prob is not None:
        try:
            out["roc_auc_macro"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro",
                labels=labels))
        except Exception:
            pass
        idx = np.array([labels.index(v) for v in y_true])
        for i, lab in enumerate(labels):
            try:
                out[f"auc::{lab}"] = float(roc_auc_score(
                    (idx == i).astype(int), y_prob[:, i]))
            except Exception:
                pass
        out["brier"] = multiclass_brier(idx, y_prob, len(labels))
        conf = y_prob.max(1)
        out["ece"] = expected_calibration_error(
            conf, (np.asarray(y_pred) == np.asarray(y_true)).astype(float))
        out["mean_confidence"] = float(conf.mean())
    return out


def bootstrap_all(y_true, y_pred, y_prob, groups, labels,
                  n: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """
    Percentile intervals for every metric, resampling whole groups.

    Resampling images would treat the ~16 correlated slices of one glioma
    patient as 16 independent observations and return intervals far too
    narrow -- the defect in the original manuscript's normal-approximation
    bounds.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    acc: dict[str, list[float]] = {}
    for _ in range(n):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in take])
        try:
            b = metric_bundle(y_true[sel], y_pred[sel],
                              None if y_prob is None else y_prob[sel], labels)
        except Exception:
            continue
        for k, v in b.items():
            acc.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 100 * alpha / 2)),
                float(np.percentile(v, 100 * (1 - alpha / 2)))]
            for k, v in acc.items() if v}


def paired_group_bootstrap(y_true, pred_a, pred_b, groups, fn,
                           n=2000, seed=0, alpha=0.05):
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
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def confusion_frame(y_true, y_pred, labels) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                        index=labels, columns=labels)


# =============================================================================
# Data
# =============================================================================

def load_data(config: str, split_mode: str, seed: int, n_folds: int):
    X = np.load(FEATURES / f"{config}_frozen.npy")
    df = pd.read_csv(FEATURES / f"{config}_index.csv")

    if split_mode == "grouped":
        return X, df, pd.read_csv(SPLITS / f"splits_{config}_inner.csv")

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


def inner_cv_for_fold(train_df, inner, k):
    sub = inner[inner.outer_fold == k]
    fold_of = dict(zip(sub.path, sub.inner_fold))
    assign = train_df.path.map(fold_of).to_numpy()
    if pd.isna(assign).any():
        raise SystemExit("some training rows have no inner-fold assignment")
    return [(np.where(assign != m)[0], np.where(assign == m)[0])
            for m in sorted(set(assign))]


# =============================================================================

def run(config, models, split_mode, seed, n_folds, tag, n_boot):
    X, df, inner = load_data(config, split_mode, seed, n_folds)
    labels = sorted(df.label.unique())
    y = df.label.to_numpy()
    groups = df.group_id.to_numpy()
    folds = df.outer_fold.to_numpy()

    # Naming matters for interpretation: only the Figshare subset has genuine
    # patient identifiers. Elsewhere a "group" is a near-duplicate cluster
    # standing in for a patient.
    unit = ("patient" if (df.patient_id.fillna("") != "").all() else "group")

    run_id = tag or f"{config}_{split_mode}"
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> run_id={run_id}  {len(df)} images, "
          f"{df.group_id.nunique()} {unit}s, {len(labels)} classes")

    all_metrics, per_fold = {}, []
    preds = {"path": df.path, "y_true": y, "group_id": groups,
             "outer_fold": folds}

    for name in models:
        print(f"\n--- {name}")
        oof_pred = np.empty(len(df), dtype=object)
        oof_prob = np.zeros((len(df), len(labels)))
        chosen, t0 = [], time.time()

        for k in range(n_folds):
            tr, te = folds != k, folds == k
            cv = inner_cv_for_fold(df[tr].reset_index(drop=True), inner, k)
            est, params = fit_tuned(name, X[tr], y[tr], cv, seed)
            chosen.append({kk.replace("clf__", ""): vv for kk, vv in params.items()})

            oof_pred[te] = est.predict(X[te])
            prob = est.predict_proba(X[te])
            col = [list(est.classes_).index(l) for l in labels]
            oof_prob[te] = prob[:, col]

            fm = metric_bundle(y[te], oof_pred[te], oof_prob[te], labels)
            per_fold.append({"model": name, "fold": k, "n": int(te.sum()),
                             **{kk: vv for kk, vv in fm.items()
                                if "::" not in kk}})
            print(f"    fold {k}: acc {fm['accuracy']:.4f}  "
                  f"bal {fm['balanced_accuracy']:.4f}  {params}")

        elapsed = time.time() - t0
        oof_pred = oof_pred.astype(str)

        m = metric_bundle(y, oof_pred, oof_prob, labels)
        ci = bootstrap_all(y, oof_pred, oof_prob, groups, labels, n_boot, seed)
        entry = {"point": m, "ci95": ci,
                 "confusion_matrix": confusion_frame(y, oof_pred, labels).values.tolist(),
                 "labels": labels,
                 "support": {l: int((y == l).sum()) for l in labels},
                 "bootstrap_unit": unit,
                 "fit_predict_seconds": round(elapsed, 1),
                 "chosen_hyperparameters_per_fold": chosen}
        all_metrics[name] = entry
        preds[f"pred_{name}"] = oof_pred
        for i, lab in enumerate(labels):
            preds[f"prob_{name}_{lab}"] = oof_prob[:, i]

        a, b = ci.get("accuracy", [np.nan, np.nan])
        print(f"    pooled: acc {m['accuracy']:.4f} [{a:.4f}, {b:.4f}]  "
              f"bal {m['balanced_accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  "
              f"AUC {m.get('roc_auc_macro', float('nan')):.4f}  "
              f"Brier {m.get('brier', float('nan')):.4f}  "
              f"ECE {m.get('ece', float('nan')):.4f}  {elapsed:.0f}s")

    # -- pairwise comparisons --------------------------------------------------
    print("\n--- pairwise comparisons")
    comparisons = []
    for a, b in combinations(models, 2):
        pa, pb = preds[f"pred_{a}"], preds[f"pred_{b}"]
        ca, cb = pa == y, pb == y
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        res = mcnemar(np.array([[int((ca & cb).sum()), n10],
                                [n01, int((~ca & ~cb).sum())]]), exact=True)
        d, lo, hi, pboot = paired_group_bootstrap(
            y, pa, pb, groups, accuracy_score, n_boot, seed)
        comparisons.append({"model_a": a, "model_b": b,
                            "a_only_correct": n10, "b_only_correct": n01,
                            "mcnemar_exact_p": float(res.pvalue),
                            "acc_diff": d, "acc_diff_ci95": [lo, hi],
                            "bootstrap_p": pboot})
        print(f"    {a} vs {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"McNemar p={res.pvalue:.4f}  bootstrap p={pboot:.4f}")

    order = np.argsort([c["mcnemar_exact_p"] for c in comparisons])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(running,
                           comparisons[i]["mcnemar_exact_p"] * (len(comparisons) - rank)))
        comparisons[i]["mcnemar_p_holm"] = float(adj)
        running = adj

    print(f"\n  NOTE: McNemar assumes independent observations, which "
          f"{df.groupby('group_id').size().mean():.1f} correlated images per "
          f"{unit}\n  violates. The paired {unit}-level bootstrap is the "
          f"primary test; McNemar is\n  reported because it was requested. "
          f"Where they disagree, the bootstrap governs.")

    # -- write -----------------------------------------------------------------
    payload = {"run_id": run_id, "config": config, "split_mode": split_mode,
               "seed": seed, "n_folds": n_folds, "n_images": len(df),
               "n_groups": int(df.group_id.nunique()), "bootstrap_unit": unit,
               "labels": labels,
               "class_counts": df.label.value_counts().to_dict(),
               "models": all_metrics, "comparisons": comparisons}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)

    pf = pd.DataFrame(per_fold)
    pf.to_csv(out_dir / "per_fold.csv", index=False)

    rows = []
    for k, v in all_metrics.items():
        r = {"model": k}
        for mk in ["accuracy", "balanced_accuracy", "macro_f1",
                   "roc_auc_macro", "brier", "ece"]:
            r[mk] = v["point"].get(mk, np.nan)
            if mk in v["ci95"]:
                r[f"{mk}_lo"], r[f"{mk}_hi"] = v["ci95"][mk]
        r["seconds"] = v["fit_predict_seconds"]
        rows.append(r)
    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\n" + summary.round(4).to_string(index=False))

    print("\n  between-fold spread (a single random split lands anywhere here):")
    print(pf.groupby("model").accuracy.agg(["min", "mean", "max", "std"])
            .round(4).to_string())

    for name, v in all_metrics.items():
        print(f"\n  per-class -- {name}")
        t = pd.DataFrame({
            met: {l: v["point"].get(f"{met}::{l}", np.nan) for l in labels}
            for met in ["precision", "recall", "specificity", "f1", "auc"]})
        t["support"] = pd.Series(v["support"])
        print(t.round(3).to_string())
        print("  confusion (rows true, cols predicted):")
        print(pd.DataFrame(v["confusion_matrix"], index=labels,
                           columns=labels).to_string())

    print(f"\n  wrote {out_dir}/")


def compare_runs(dirs):
    rows = []
    for d in dirs:
        p = Path(d)
        if not (p / "metrics.json").exists():
            p = RUNS / d
        meta = json.loads((p / "metrics.json").read_text())
        for name, m in meta["models"].items():
            pt, ci = m["point"], m["ci95"]
            rows.append({"run": meta["run_id"], "split": meta["split_mode"],
                         "model": name, "accuracy": pt["accuracy"],
                         "ci_lo": ci.get("accuracy", [np.nan] * 2)[0],
                         "ci_hi": ci.get("accuracy", [np.nan] * 2)[1],
                         "balanced_accuracy": pt["balanced_accuracy"],
                         "macro_f1": pt["macro_f1"]})
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    if t.split.nunique() > 1:
        piv = t.pivot_table(index="model", columns="split", values="accuracy")
        if {"grouped", "image"} <= set(piv.columns):
            piv["leakage_inflation"] = piv["image"] - piv["grouped"]
            print("\nAccuracy inflation attributable to image-level splitting:")
            print(piv.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="main")
    ap.add_argument("--models", nargs="+",
                    default=["svm", "knn", "rf", "mlp", "logreg"])
    ap.add_argument("--split-mode", choices=["grouped", "image"], default="grouped")
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