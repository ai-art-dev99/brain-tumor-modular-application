#!/usr/bin/env python3
"""
report_tables.py -- build every manuscript and supplementary table from the
stored artefacts.

Nothing here recomputes a result. Everything is read from manifest CSVs and
from each run's metrics.json, so the tables cannot drift from the numbers that
were actually produced. Run it last, after all experiments, and again after
any rerun.

TABLES PRODUCED
---------------
  table1_sources.csv        source provenance and acquisition metadata, with
                            explicit "NR" where the original repository did
                            not report a field. Reviewer point 2 asks for
                            acquisition parameters; where they do not exist,
                            the correct answer is to say so rather than to
                            infer them from JPEG appearance.
  table2_partitions.csv     exact class counts per fold, in images and in
                            independent units, plus the ratio between them.
  table3_augmentation.csv   augmentation accounting. Stored augmented files is
                            zero: transforms are applied on the fly to training
                            observations only, so no augmented image can reach
                            a validation or test partition.
  table4_main_results.csv   headline comparison across runs.
  table5_leakage.csv        grouped versus image-level, per model.
  supp_perclass_ci.csv      every per-class metric with its 95% bootstrap
                            interval, for every model in every run.
  supp_confusion_<run>_<model>.csv   real-count confusion matrices.
  supp_perfold.csv          fold-level results, so between-fold spread is
                            visible rather than hidden inside a pooled mean.
  supp_comparisons.csv      all pairwise tests, with Holm-adjusted p-values on
                            both McNemar and the bootstrap, and the
                            prespecified comparison flagged.

PRESPECIFIED COMPARISON
-----------------------
The editor asked specifically whether the SVM hybrid improves on the standalone
CNN. That one comparison is prespecified and is reported unadjusted. The
remaining pairwise tests are exploratory and carry Holm-adjusted values; a
reviewer who sees fifteen uncorrected p-values will ask about multiplicity.

Usage
-----
    python report_tables.py --runs main_finetuned_v2 figshare_finetuned_v2 \\
        main_naive_ft_v2 main_grouped_v2
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path("/workspace/data/manifest")
SPLITS = MANIFEST / "splits"
RUNS = Path("/workspace/outputs/runs")
OUT = Path("/workspace/outputs/tables")

# Acquisition fields as published by each repository. NR = not reported by the
# original source. These are not inferred from the images.
SOURCE_META = {
    "figshare": {"sequence": "T1-weighted", "contrast": "Contrast-enhanced",
                 "plane": "Axial, coronal, sagittal", "scanner": "NR",
                 "native_size": "512x512", "pid_available": "Yes",
                 "mask_available": "Yes",
                 "reference": "Cheng et al., Figshare 1512427"},
    "sartaj":   {"sequence": "NR", "contrast": "NR", "plane": "NR",
                 "scanner": "NR", "native_size": "Variable",
                 "pid_available": "No", "mask_available": "No",
                 "reference": "Bhuvaji et al., Kaggle"},
    "br35h":    {"sequence": "NR (mixed T2/FLAIR on inspection)",
                 "contrast": "NR", "plane": "NR", "scanner": "NR",
                 "native_size": "Variable", "pid_available": "No",
                 "mask_available": "No", "reference": "Hamada, Kaggle Br35H"},
}

PRESPECIFIED = ("cnn", "cnn_svm")


def _load(run: str) -> dict | None:
    p = RUNS / run / "metrics.json"
    if not p.exists():
        print(f"  skipping {run}: no metrics.json")
        return None
    return json.loads(p.read_text())


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    n, out, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(running, pvals[i] * (n - rank)))
        out[i] = adj
        running = adj
    return out


# =============================================================================

def table1_sources() -> pd.DataFrame:
    ds = pd.read_csv(MANIFEST / "dataset.csv")
    ds["patient_id"] = ds.patient_id.fillna("")
    files = pd.read_csv(MANIFEST / "files_index.csv")

    rows = []
    for src, g in ds.groupby("source"):
        meta = SOURCE_META.get(src, {})
        raw = files[files.source == src]
        dims = (raw.width.astype(str) + "x" + raw.height.astype(str)).nunique() \
            if len(raw) else np.nan
        rows.append({
            "source": src,
            "images_raw": int(len(raw)) if len(raw) else np.nan,
            "images_after_dedup": int(len(g)),
            "distinct_dimensions_raw": dims,
            "patients": (int(g.patient_id[g.patient_id != ""].nunique())
                         if (g.patient_id != "").any() else "NR"),
            "leakage_control_groups": int(g.group_id.nunique()),
            "classes": ", ".join(sorted(g.label.unique())),
            **{k: meta.get(k, "NR") for k in
               ["sequence", "contrast", "plane", "scanner", "native_size",
                "pid_available", "mask_available", "reference"]},
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"source": "TOTAL",
                       "images_after_dedup": int(len(ds)),
                       "leakage_control_groups": int(ds.group_id.nunique()),
                       **{c: "" for c in df.columns
                          if c not in ("source", "images_after_dedup",
                                       "leakage_control_groups")}}
    return df


def table2_partitions(configs: list[str]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        f = SPLITS / f"splits_{cfg}_outer.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for (fold, lab), g in d.groupby(["outer_fold", "label"]):
            rows.append({"config": cfg, "fold": int(fold), "class": lab,
                         "images": len(g),
                         "independent_units": g.group_id.nunique(),
                         "images_per_unit": round(len(g) / g.group_id.nunique(), 2)})
    return pd.DataFrame(rows)


def table3_augmentation(configs: list[str], runs: list[str]) -> pd.DataFrame:
    """
    Reviewer point 4 asks for class-wise counts before and after augmentation
    in every partition. In this pipeline augmentation is applied on the fly to
    training observations only and nothing is written to disk, so the counts
    before and after are identical by construction and no augmented image can
    reach a validation or test partition. Stating that explicitly is the
    answer; a table of doubled counts would misdescribe what was done.
    """
    aug = "NR"
    for r in runs:
        m = _load(r)
        if m and m.get("augmentation"):
            aug = m["augmentation"]
            break
    rows = []
    for cfg in configs:
        f = SPLITS / f"splits_{cfg}_outer.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        inner = SPLITS / f"splits_{cfg}_inner.csv"
        idf = pd.read_csv(inner) if inner.exists() else None
        for fold in sorted(d.outer_fold.unique()):
            tr, te = d[d.outer_fold != fold], d[d.outer_fold == fold]
            fit = tr
            if idf is not None:
                sub = idf[idf.outer_fold == fold]
                mp = dict(zip(sub.path, sub.inner_fold))
                fit = tr[tr.path.map(mp) != 0]
            for part, sel, augmented in [("train_fit", fit, True),
                                         ("train_earlystop", tr[~tr.path.isin(fit.path)], False),
                                         ("test", te, False)]:
                for lab, g in sel.groupby("label"):
                    rows.append({
                        "config": cfg, "fold": int(fold), "partition": part,
                        "class": lab,
                        "images_before_augmentation": len(g),
                        "images_after_augmentation": len(g),
                        "stored_augmented_files": 0,
                        "augmentation_applied": "on-the-fly, stochastic"
                                                if augmented else "none",
                    })
    out = pd.DataFrame(rows)
    out.attrs["augmentation_spec"] = aug
    return out


def table4_main(runs: list[str]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = _load(r)
        if not m:
            continue
        for name, v in m["models"].items():
            pt, ci = v["point"], v.get("ci95", {})
            rows.append({
                "run": m["run_id"], "config": m["config"],
                "split": m.get("split_mode", "grouped"),
                "unit": m.get("bootstrap_unit", "group"),
                "model": name,
                "n_images": m["n_images"], "n_units": m["n_groups"],
                "accuracy": pt["accuracy"],
                "acc_lo": ci.get("accuracy", [np.nan] * 2)[0],
                "acc_hi": ci.get("accuracy", [np.nan] * 2)[1],
                "balanced_accuracy": pt["balanced_accuracy"],
                "bal_lo": ci.get("balanced_accuracy", [np.nan] * 2)[0],
                "bal_hi": ci.get("balanced_accuracy", [np.nan] * 2)[1],
                "macro_f1": pt["macro_f1"],
                "macro_auc": pt.get("roc_auc_macro", np.nan),
                "brier": pt.get("brier", np.nan),
                "ece": pt.get("ece", np.nan),
            })
    return pd.DataFrame(rows)


def table5_leakage(main_df: pd.DataFrame) -> pd.DataFrame:
    piv = main_df.pivot_table(index="model", columns="split", values="accuracy")
    if not {"grouped", "image"} <= set(piv.columns):
        return pd.DataFrame()
    piv = piv.rename(columns={"grouped": "leakage_controlled",
                              "image": "naive_image_level"})
    piv["apparent_inflation_pp"] = (
        100 * (piv.naive_image_level - piv.leakage_controlled)).round(2)
    return piv.reset_index()


def supp_perclass(runs: list[str]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = _load(r)
        if not m:
            continue
        labels = m["labels"]
        for name, v in m["models"].items():
            pt, ci = v["point"], v.get("ci95", {})
            for lab in labels:
                rec = {"run": m["run_id"], "model": name, "class": lab,
                       "support": v.get("support", {}).get(lab, np.nan)}
                for met in ["precision", "recall", "specificity", "f1", "auc"]:
                    key = f"{met}::{lab}"
                    rec[met] = pt.get(key, np.nan)
                    lo, hi = ci.get(key, [np.nan, np.nan])
                    rec[f"{met}_lo"], rec[f"{met}_hi"] = lo, hi
                rows.append(rec)
    return pd.DataFrame(rows)


def supp_confusions(runs: list[str]) -> None:
    for r in runs:
        m = _load(r)
        if not m:
            continue
        labels = m["labels"]
        for name, v in m["models"].items():
            cm = v.get("confusion_matrix")
            if cm is None:
                continue
            d = pd.DataFrame(cm, index=labels, columns=labels)
            d.index.name = "true"
            d.to_csv(OUT / f"supp_confusion_{m['run_id']}_{name}.csv")


def supp_comparisons(runs: list[str]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = _load(r)
        if not m or "comparisons" not in m:
            continue
        cmps = m["comparisons"]
        pre = [ (c["model_a"], c["model_b"]) == PRESPECIFIED or
                (c["model_b"], c["model_a"]) == PRESPECIFIED for c in cmps ]
        expl = [i for i, f in enumerate(pre) if not f]
        # Holm is applied to the exploratory family only; the prespecified
        # comparison is reported unadjusted because it was stated in advance.
        mh = holm([cmps[i]["mcnemar_exact_p"] for i in expl]) if expl else []
        bh = holm([cmps[i]["bootstrap_p"] for i in expl]) if expl else []
        for j, i in enumerate(expl):
            cmps[i]["mcnemar_p_holm"] = mh[j]
            cmps[i]["bootstrap_p_holm"] = bh[j]
        for i, c in enumerate(cmps):
            rows.append({"run": m["run_id"], "prespecified": pre[i],
                         "model_a": c["model_a"], "model_b": c["model_b"],
                         "acc_diff": c["acc_diff"],
                         "ci_lo": c["acc_diff_ci95"][0],
                         "ci_hi": c["acc_diff_ci95"][1],
                         "bootstrap_p": c["bootstrap_p"],
                         "bootstrap_p_holm": c.get("bootstrap_p_holm", np.nan),
                         "mcnemar_p": c["mcnemar_exact_p"],
                         "mcnemar_p_holm": c.get("mcnemar_p_holm", np.nan),
                         "a_only_correct": c["a_only_correct"],
                         "b_only_correct": c["b_only_correct"]})
    return pd.DataFrame(rows)


def supp_perfold(runs: list[str]) -> pd.DataFrame:
    frames = []
    for r in runs:
        f = RUNS / r / "per_fold.csv"
        if f.exists():
            d = pd.read_csv(f)
            d.insert(0, "run", r)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+",
                    default=["main", "figshare_only", "main_imagelevel"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t1 = table1_sources()
    t1.to_csv(OUT / "table1_sources.csv", index=False)
    print("== Table 1: sources\n" + t1.to_string(index=False) + "\n")

    t2 = table2_partitions(a.configs)
    t2.to_csv(OUT / "table2_partitions.csv", index=False)

    t3 = table3_augmentation(a.configs, a.runs)
    t3.to_csv(OUT / "table3_augmentation.csv", index=False)
    print(f"== Table 3: augmentation spec\n  {t3.attrs.get('augmentation_spec')}")
    print(f"  stored augmented files: 0 (transforms applied on the fly to "
          f"training observations only)\n")

    t4 = table4_main(a.runs)
    t4.to_csv(OUT / "table4_main_results.csv", index=False)
    print("== Table 4: main results")
    print(t4[["run", "split", "unit", "model", "accuracy", "acc_lo", "acc_hi",
              "balanced_accuracy", "macro_f1", "macro_auc", "ece"]]
          .round(4).to_string(index=False) + "\n")

    t5 = table5_leakage(t4)
    if len(t5):
        t5.to_csv(OUT / "table5_leakage.csv", index=False)
        print("== Table 5: leakage inflation")
        print(t5.round(4).to_string(index=False) + "\n")

    sp = supp_perclass(a.runs)
    sp.to_csv(OUT / "supp_perclass_ci.csv", index=False)
    print(f"== Supplementary: per-class metrics with 95% CI "
          f"({len(sp)} rows)")
    if len(sp):
        ex = sp[sp.run == a.runs[0]]
        print(ex[["model", "class", "support", "recall", "recall_lo",
                  "recall_hi", "f1", "f1_lo", "f1_hi"]].round(3)
              .to_string(index=False) + "\n")

    supp_confusions(a.runs)
    sc = supp_comparisons(a.runs)
    sc.to_csv(OUT / "supp_comparisons.csv", index=False)
    if len(sc):
        print("== Comparisons (prespecified reported unadjusted, "
              "exploratory Holm-adjusted)")
        print(sc[sc.prespecified][["run", "model_a", "model_b", "acc_diff",
                                   "ci_lo", "ci_hi", "bootstrap_p",
                                   "mcnemar_p"]].round(4).to_string(index=False))
        print()

    pf = supp_perfold(a.runs)
    if len(pf):
        pf.to_csv(OUT / "supp_perfold.csv", index=False)
        print("== Between-fold spread")
        print(pf.groupby(["run", "model"]).accuracy
                .agg(["min", "mean", "max", "std"]).round(4).to_string())

    print(f"\n  wrote {OUT}/")


if __name__ == "__main__":
    main()