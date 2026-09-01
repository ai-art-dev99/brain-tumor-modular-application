#!/usr/bin/env python3
"""
eval_external.py (v2) -- leakage-associated performance inflation measured on a
third-party release.

THE DESIGN
----------
One published test partition, one set of frozen models, one inference run.
Images are stratified by their relationship to our development data:

  direct       matches the 3,813 images the models were actually fitted on
  source_only  matches the source repositories but not the fitted set
  clean        no match to the source repositories at the pre-specified
               threshold
  strict_clean a subset of `clean` at nearest distance >= 12, i.e. beyond the
               empty region of the distance distribution

Everything else is held constant: same publisher, same preprocessing, same
weights, same code path, same session. The contrast between strata is
therefore a direct empirical estimate of how much overlap with development
data is worth, obtained without reference to our own splits.

WHAT THE GAP IS AND IS NOT
--------------------------
It is leakage-ASSOCIATED inflation, not inflation caused by leakage. Overlap
status is not randomly assigned: it correlates with class, with image quality,
and possibly with acquisition subpopulation. Two features of this design
narrow the alternatives without eliminating them.

  1. Class mixture differs between strata, so balanced accuracy is primary and
     the per-class gaps are reported individually. A gap present in all four
     classes cannot be explained by class composition.
  2. `strict_clean` excludes anything within 11 bits of a development image,
     which tests the objection that unmatched images are merely
     harder-to-match versions of the same scans. If `clean` and `strict_clean`
     agree, that objection does not hold.

STATISTICS
----------
The two strata are different observation sets, so a paired test does not
apply. The difference in balanced accuracy is bootstrapped with resampling
within class within stratum, which holds the class mixture fixed across
iterations.

PRE-SPECIFICATION
-----------------
Primary analysis: `direct` versus `clean`, balanced accuracy, threshold <= 2.
The distance-bin analysis at the end is exploratory and labelled as such; it
selects thresholds after seeing data and must not be reported as confirmatory.

Usage
-----
    python eval_external.py --run main_finetuned_v2 \\
        --manifest /workspace/data/external/bdneuro_v7_manifest.csv --split test
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_eval import bootstrap_all, confusion_frame, fit_tuned, metric_bundle

SPLITS = Path("/workspace/data/manifest/splits")
RUNS = Path("/workspace/outputs/runs")
OUT = Path("/workspace/outputs/external")

STRICT_CLEAN_MIN = 12       # beyond the empty region of the distribution


class Images(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), i


@torch.no_grad()
def forward_all(model, paths, tf, device, n_classes, dim, bs, workers):
    dl = DataLoader(Images(paths, tf), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=True)
    lg = torch.zeros(len(paths), n_classes)
    ft = np.empty((len(paths), dim), dtype=np.float32)
    for xb, idx in dl:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device == "cuda"):
            f = model.forward_features(xb)
            pooled = model.forward_head(f, pre_logits=True)
            out = model.forward_head(f)
        lg[idx] = out.float().cpu()
        ft[idx.numpy()] = pooled.float().cpu().numpy()
    return lg, ft


# =============================================================================
# Statistics
# =============================================================================

def fmt_p(p: float, n_boot: int) -> str:
    """A bootstrap cannot resolve p below 1/n; printing 0.0000 overstates it."""
    floor = 1.0 / n_boot
    return f"{p:.4f}" if p >= floor else f"<{floor:.4f}"


def balanced_acc(y, p, labels) -> float:
    return float(recall_score(y, p, labels=labels, average="macro",
                              zero_division=0))


def stratified_delta(yA, pA, yB, pB, labels, n=4000, seed=42, alpha=0.05):
    """
    Bootstrap the difference in balanced accuracy between two independent
    strata, resampling within class within stratum. Holding class sizes fixed
    keeps the difference from absorbing the strata's different class mixtures.
    """
    rng = np.random.default_rng(seed)
    idxA = {l: np.where(yA == l)[0] for l in labels}
    idxB = {l: np.where(yB == l)[0] for l in labels}
    usable = [l for l in labels if len(idxA[l]) and len(idxB[l])]
    if len(usable) < len(labels):
        missing = set(labels) - set(usable)
        print(f"    note: classes absent from one stratum, excluded from the "
              f"balanced figure: {', '.join(sorted(missing))}")

    out = []
    for _ in range(n):
        rA, rB = [], []
        for l in usable:
            a = rng.choice(idxA[l], len(idxA[l]), replace=True)
            b = rng.choice(idxB[l], len(idxB[l]), replace=True)
            rA.append((pA[a] == l).mean())
            rB.append((pB[b] == l).mean())
        out.append(np.mean(rA) - np.mean(rB))
    d = np.array(out)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def per_class_recall_ci(y, p, labels, n=4000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    res = {}
    for l in labels:
        idx = np.where(y == l)[0]
        if not len(idx):
            res[l] = (np.nan, np.nan, np.nan, 0)
            continue
        point = float((p[idx] == l).mean())
        draws = [(p[rng.choice(idx, len(idx), replace=True)] == l).mean()
                 for _ in range(n)]
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        res[l] = (point, float(lo), float(hi), len(idx))
    return res


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main_finetuned_v2")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--heads", nargs="*",
                    default=["svm", "knn", "rf", "mlp", "logreg"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RUNS / a.run / "metrics.json").read_text())
    labels = meta["labels"]
    dev = pd.read_csv(SPLITS / f"splits_{meta['config']}_outer.csv")
    inner = pd.read_csv(SPLITS / f"splits_{meta['config']}_inner.csv")

    ext = pd.read_csv(a.manifest)
    ext = ext[ext.label.isin(labels)]
    if a.split != "all" and "published_split" in ext.columns:
        ext = ext[ext.published_split == a.split]
    ext = ext.reset_index(drop=True)

    for c in ["trained_overlap", "sources_overlap", "sources_distance"]:
        if c not in ext.columns:
            raise SystemExit(f"manifest lacks '{c}'; rerun audit_external.py")

    # -- strata ---------------------------------------------------------------
    tr_ov = ext.trained_overlap.astype(bool).to_numpy()
    sr_ov = ext.sources_overlap.astype(bool).to_numpy()
    dist = ext.sources_distance.to_numpy()
    stratum = np.where(tr_ov, "direct",
                       np.where(sr_ov, "source_only", "clean"))
    ext["stratum"] = stratum
    strict = (~sr_ov) & (dist >= STRICT_CLEAN_MIN)

    tag = a.tag or f"{Path(a.manifest).stem}_{a.split}"
    print(f"==> {len(ext)} images from partition '{a.split}'")
    print("\n  strata by class:")
    print(pd.crosstab(ext.label, ext.stratum, margins=True).to_string())
    print(f"\n  strict_clean (nearest source distance >= {STRICT_CLEAN_MIN}): "
          f"{int(strict.sum())}")
    so = ext.label[stratum == "source_only"]
    if len(so) and so.nunique() < len(labels):
        print(f"\n  NOTE: the source_only stratum contains only "
              f"{', '.join(sorted(so.unique()))}.")
        print("  Balanced accuracy over the full label set assigns zero recall")
        print("  to absent classes, so its value is not comparable with the")
        print("  four-class strata. Report it descriptively, per class, and")
        print("  build no claim on it.")
    print("\n  Class mixture differs between strata, so balanced accuracy is")
    print("  the primary metric and per-class gaps are reported individually.")

    # -- distance distribution of the clean stratum ---------------------------
    cd = dist[stratum == "clean"]
    if len(cd):
        print("\n  nearest-source distance within the clean stratum:")
        print(f"    min {cd.min()}  q25 {np.percentile(cd, 25):.0f}  "
              f"median {np.median(cd):.0f}  q75 {np.percentile(cd, 75):.0f}  "
              f"max {cd.max()}")
        for lo, hi in [(3, 5), (6, 11), (12, 15), (16, 99)]:
            n = int(((cd >= lo) & (cd <= hi)).sum())
            print(f"    {lo:2d}-{hi:<3d}: {n:5d} ({100 * n / len(cd):5.1f}%)")
        far = int((cd >= STRICT_CLEAN_MIN).sum())
        print(f"  {100 * far / len(cd):.1f}% of clean images sit at "
              f"{STRICT_CLEAN_MIN} bits or beyond. The larger that share,")
        print("  the weaker the objection that they are merely near-misses of")
        print("  matching. Quote the percentage, not an impression of it.")

    # -- inference ------------------------------------------------------------
    tmp = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(tmp)
    size, dim = cfg["input_size"][1], tmp.num_features
    del tmp
    tf = transforms.Compose([transforms.Resize((size, size)),
                             transforms.ToTensor(),
                             transforms.Normalize(cfg["mean"], cfg["std"])])

    n_folds = dev.outer_fold.nunique()
    names = ["cnn"] + [f"cnn_{h}" for h in a.heads]
    prob = {n: np.zeros((len(ext), len(labels))) for n in names}

    for k in range(n_folds):
        ck = RUNS / a.run / f"backbone_fold{k}.pt"
        if not ck.exists():
            raise SystemExit(f"{ck} missing; rerun train_cnn.py with "
                             f"--save-checkpoints.")
        print(f"\n--- fold {k}")
        model = timm.create_model("efficientnet_b0", pretrained=False,
                                  num_classes=len(labels))
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval().to(device)

        lg, Xe = forward_all(model, ext.path.tolist(), tf, device,
                             len(labels), dim, a.batch_size, a.workers)
        prob["cnn"] += torch.softmax(lg, 1).numpy() / n_folds

        if a.heads:
            tr = dev[dev.outer_fold != k].reset_index(drop=True)
            sub = inner[inner.outer_fold == k]
            tr["inner"] = tr.path.map(dict(zip(sub.path, sub.inner_fold)))
            _, Xtr = forward_all(model, tr.path.tolist(), tf, device,
                                 len(labels), dim, a.batch_size, a.workers)
            asg = tr.inner.to_numpy()
            cv = [(np.where(asg != m)[0], np.where(asg == m)[0])
                  for m in sorted(set(asg))]
            for h in a.heads:
                est, params = fit_tuned(h, Xtr, tr.label.to_numpy(), cv, a.seed)
                p = est.predict_proba(Xe)
                col = [list(est.classes_).index(l) for l in labels]
                prob[f"cnn_{h}"] += p[:, col] / n_folds
                print(f"    cnn_{h} head refitted {params}")
        del model
        torch.cuda.empty_cache()

    # -- results per stratum ---------------------------------------------------
    y = ext.label.to_numpy()
    groups = (ext.group_id.to_numpy() if "group_id" in ext.columns
              else np.arange(len(ext)).astype(str))
    masks = {"direct": stratum == "direct",
             "source_only": stratum == "source_only",
             "clean": stratum == "clean",
             "strict_clean": strict}

    results, table, preds = {}, [], {"path": ext.path, "label": y,
                                     "stratum": stratum,
                                     "sources_distance": dist,
                                     "group_id": groups}
    print("\n" + "=" * 82)
    print("Results by stratum (balanced accuracy is primary)")
    print("=" * 82)
    for n in names:
        pr = np.array(labels)[prob[n].argmax(1)]
        preds[f"pred_{n}"] = pr
        for i, l in enumerate(labels):
            preds[f"prob_{n}_{l}"] = prob[n][:, i]
        entry = {}
        for s, m in masks.items():
            if m.sum() < 20:
                continue
            mb = metric_bundle(y[m], pr[m], prob[n][m], labels)
            ci = bootstrap_all(y[m], pr[m], prob[n][m], groups[m], labels,
                               a.bootstrap // 2, a.seed)
            lo, hi = ci.get("balanced_accuracy", [np.nan, np.nan])
            conf = prob[n][m].max(1)
            entry[s] = {"n": int(m.sum()), "point": mb, "ci95": ci,
                        "mean_confidence": float(conf.mean()),
                        "confusion_matrix":
                            confusion_frame(y[m], pr[m], labels).values.tolist()}
            print(f"  {n:<12} {s:<13} n={int(m.sum()):4d}  "
                  f"BA {mb['balanced_accuracy']:.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"acc {mb['accuracy']:.4f}  F1 {mb['macro_f1']:.4f}  "
                  f"conf {conf.mean():.3f}  ECE "
                  f"{mb.get('ece', float('nan')):.3f}")

        # post-hoc contrast against the strictly separated subset
        if {"direct", "strict_clean"} <= set(entry):
            A, S = masks["direct"], masks["strict_clean"]
            d2, lo2, hi2, p2 = stratified_delta(y[A], pr[A], y[S], pr[S],
                                                labels, a.bootstrap, a.seed)
            entry["delta_direct_minus_strict_clean"] = {
                "balanced_accuracy_diff": d2, "ci95": [lo2, hi2], "p": p2,
                "prespecified": False}
            print(f"  {n:<12} {'DELTA(post)':<13} direct - strict_clean: "
                  f"{100 * d2:+.2f} pp  95% CI [{100 * lo2:+.2f}, "
                  f"{100 * hi2:+.2f}]  p={fmt_p(p2, a.bootstrap)}")

        # primary contrast
        if {"direct", "clean"} <= set(entry):
            A, B = masks["direct"], masks["clean"]
            d, lo, hi, p = stratified_delta(y[A], pr[A], y[B], pr[B], labels,
                                            a.bootstrap, a.seed)
            entry["delta_direct_minus_clean"] = {
                "balanced_accuracy_diff": d, "ci95": [lo, hi], "p": p}
            print(f"  {n:<12} {'DELTA':<13} direct - clean: "
                  f"{100 * d:+.2f} pp  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]"
                  f"  p={fmt_p(p, a.bootstrap)}")
            table.append({"model": n, "delta_pp": 100 * d,
                          "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi, "p": p,
                          "direct_BA": entry["direct"]["point"]["balanced_accuracy"],
                          "clean_BA": entry["clean"]["point"]["balanced_accuracy"],
                          "strict_clean_BA":
                              entry.get("strict_clean", {}).get("point", {})
                              .get("balanced_accuracy", np.nan),
                          "delta_strict_pp": 100 * entry.get(
                              "delta_direct_minus_strict_clean", {}).get(
                              "balanced_accuracy_diff", np.nan),
                          "delta_strict_p": entry.get(
                              "delta_direct_minus_strict_clean", {}).get(
                              "p", np.nan)})

        # per-class recall, the check that class mixture cannot explain the gap
        if {"direct", "clean"} <= set(entry):
            rc = {}
            for s in ["direct", "clean"]:
                m = masks[s]
                rc[s] = per_class_recall_ci(y[m], pr[m], labels,
                                            a.bootstrap // 2, a.seed)
            entry["per_class_recall"] = rc
            if n in ("cnn", "cnn_svm"):
                print(f"\n    per-class recall, {n}:")
                print(f"      {'class':<12} {'direct':>22} {'clean':>22} "
                      f"{'gap pp':>8}")
                for l in labels:
                    dp, dlo, dhi, dn = rc["direct"][l]
                    cp, clo, chi, cn = rc["clean"][l]
                    print(f"      {l:<12} {dp:.3f} [{dlo:.3f},{dhi:.3f}] n={dn:<4d}"
                          f" {cp:.3f} [{clo:.3f},{chi:.3f}] n={cn:<4d}"
                          f" {100 * (dp - cp):+7.1f}")
                print()
        results[n] = entry
        print()

    # -- headline table --------------------------------------------------------
    print("=" * 82)
    print("Leakage-associated inflation on a third-party release")
    print("=" * 82)
    t = pd.DataFrame(table)
    # Six models are compared. cnn and cnn_svm are the prespecified pair and
    # are also quoted unadjusted; the family carries Holm-adjusted values so a
    # multiplicity objection has an answer.
    order = np.argsort(t.p.to_numpy())
    adj, running = np.zeros(len(t)), 0.0
    for rank, i in enumerate(order):
        running = min(1.0, max(running, t.p.iloc[i] * (len(t) - rank)))
        adj[i] = running
    t["p_holm"] = adj
    print(t.round(4).to_string(index=False))
    print("\n  Same release, same weights, same preprocessing, one run. The gap")
    print("  is leakage-ASSOCIATED inflation: overlap status is not randomly")
    print("  assigned and correlates with class and image quality. Balanced")
    print("  accuracy and the per-class gaps address the class confound; the")
    print("  strict_clean column addresses the objection that unmatched images")
    print("  are merely harder to match.")
    print("\n  Do not describe any stratum as independent external validation.")

    # -- exploratory: performance against distance ----------------------------
    print("\n" + "=" * 82)
    print("EXPLORATORY: balanced accuracy against distance to development data")
    print("=" * 82)
    print("  Thresholds below are chosen after seeing the data. Report as")
    print("  exploratory; the confirmatory analysis is the pre-specified")
    print("  direct-versus-clean contrast above.")
    bins = [(0, 0), (1, 2), (3, 11), (12, 15), (16, 200)]
    for n in ["cnn", "cnn_svm"]:
        if n not in results:
            continue
        pr = preds[f"pred_{n}"]
        print(f"\n  {n}")
        for lo, hi in bins:
            m = (dist >= lo) & (dist <= hi)
            if m.sum() < 20:
                continue
            print(f"    distance {lo:3d}-{hi:<3d}  n={int(m.sum()):4d}  "
                  f"BA {balanced_acc(y[m], pr[m], labels):.4f}  "
                  f"conf {prob[n][m].max(1).mean():.3f}")

    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)
    t.to_csv(out_dir / "leakage_gap.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "tag": tag, "source_run": a.run, "manifest": a.manifest,
        "partition": a.split, "strict_clean_min_distance": STRICT_CLEAN_MIN,
        "n_images": len(ext),
        "stratum_counts": pd.Series(stratum).value_counts().to_dict(),
        "class_by_stratum": pd.crosstab(ext.label, ext.stratum).to_dict(),
        "labels": labels,
        "fold_combination": "mean posterior over 5 backbones",
        "primary_analysis": "direct vs clean, balanced accuracy, "
                            "stratified bootstrap",
        "models": results}, indent=2, default=str))
    print(f"\n  wrote {out_dir}/")


if __name__ == "__main__":
    main()