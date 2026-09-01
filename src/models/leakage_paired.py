#!/usr/bin/env python3
"""
leakage_paired.py -- a within-image paired test of memorisation.

THE PROBLEM WITH THE STRATIFIED COMPARISON
------------------------------------------
Comparing overlapping against non-overlapping external images holds the
publisher, preprocessing and model constant, but not the images. Overlap
status is not randomly assigned: it correlates with class, with image quality,
and possibly with acquisition subpopulation. The strict-clean sensitivity
analysis showed the gap shrinking from ~6 pp to ~2 pp, which is exactly what a
residual confound looks like.

THE DESIGN HERE
---------------
Each external image that matches a development image is scored by five
backbones. The matched development image sat in the held-out fold of exactly
one of them, so that one backbone was fitted WITHOUT it while the other four
were fitted WITH it. The comparison is therefore:

    same image, same class, same preprocessing, same architecture,
    same amount of training data, same inference code
    -- only the presence of one specific counterpart differs.

Every confound that troubles the stratified comparison is held fixed by
construction. What remains is a paired contrast over images.

WHAT IT CANNOT SHOW
-------------------
The counterpart is one image among roughly three thousand, so the expected
effect is small unless the model memorises individual examples. A null result
here is informative: it would mean the stratified gap reflects population
differences rather than memorisation of specific images, and the manuscript
should say so.

Conversely a positive result is strong: no plausible confound survives a
design in which the image is its own control.

NOTE ON GROUPS
--------------
Fold membership is looked up through the development image's leakage-control
group, since the whole group moves together across folds. Where a match points
at an image whose group spans no fold (should not happen) the pair is dropped
and counted.

Usage
-----
    python leakage_paired.py --run main_finetuned_v2 \\
        --manifest /workspace/data/external/bdneuro_v7_manifest.csv --split all
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_eval import fit_tuned

SPLITS = Path("/workspace/data/manifest/splits")
RUNS = Path("/workspace/outputs/runs")
OUT = Path("/workspace/outputs/external")


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
            lg[idx] = model.forward_head(f).float().cpu()
            ft[idx.numpy()] = model.forward_head(
                f, pre_logits=True).float().cpu().numpy()
    return lg, ft


def paired_bootstrap(seen, unseen, n=4000, seed=42, alpha=0.05):
    """
    Bootstrap the difference in accuracy between models that saw an image's
    counterpart and the model that did not, resampling images. `seen` holds
    each image's mean correctness over the four exposed backbones; `unseen`
    holds its correctness under the one unexposed backbone.
    """
    rng = np.random.default_rng(seed)
    d = seen - unseen
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean()
                      for _ in range(n)])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main_finetuned_v2")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="all",
                    help="published partition, or 'all' for maximum power; "
                         "this is a diagnostic, not a performance claim")
    ap.add_argument("--heads", nargs="*", default=["svm", "logreg"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control", action="store_true",
                    help="negative control: use NON-overlapping images and "
                         "assign each a random pseudo-counterpart fold. The "
                         "contrast must come out null; anything else means the "
                         "estimator itself is biased and the main result "
                         "cannot be trusted.")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((RUNS / a.run / "metrics.json").read_text())
    labels = meta["labels"]
    dev = pd.read_csv(SPLITS / f"splits_{meta['config']}_outer.csv")
    inner = pd.read_csv(SPLITS / f"splits_{meta['config']}_inner.csv")
    fold_of_path = dict(zip(dev.path, dev.outer_fold))

    ext = pd.read_csv(a.manifest)
    ext = ext[ext.label.isin(labels)]
    if a.split != "all" and "published_split" in ext.columns:
        ext = ext[ext.published_split == a.split]

    dropped = 0
    if a.control:
        # Non-overlapping images have no counterpart, so the fold index is
        # assigned at random. Every mechanical feature of the analysis is
        # identical to the real test -- same images scored five times, same
        # leave-one-fold-out arithmetic, same bootstrap -- but the grouping
        # variable carries no information. A non-null result here would mean
        # the contrast is driven by differences between the backbones rather
        # than by what they were trained on.
        ext = ext[~ext.sources_overlap.astype(bool)].reset_index(drop=True)
        if not len(ext):
            raise SystemExit("no non-overlapping images available for the control")
        rng = np.random.default_rng(a.seed)
        ext["counterpart_fold"] = rng.integers(0, dev.outer_fold.nunique(),
                                               len(ext))
        ext["trained_match_path"] = ""
        ext["trained_distance"] = -1
    else:
        ext = ext[ext.trained_overlap.astype(bool)].reset_index(drop=True)
        if not len(ext):
            raise SystemExit("no images overlap the fitted development set")
        # Folds are group-disjoint, so the unit held out is the counterpart's
        # whole leakage-control group, not a single image: around 16 images for
        # glioma, around 2 for no-tumour. Describe it that way when reporting.
        ext["counterpart_fold"] = ext.trained_match_path.map(fold_of_path)
        dropped = int(ext.counterpart_fold.isna().sum())
        ext = ext.dropna(subset=["counterpart_fold"]).reset_index(drop=True)
    ext["counterpart_fold"] = ext.counterpart_fold.astype(int)

    n_folds = dev.outer_fold.nunique()
    mode = "NEGATIVE CONTROL" if a.control else "paired leakage test"
    print(f"==> {mode}: {len(ext)} images"
          f"{f' ({dropped} dropped: counterpart not in the split file)' if dropped else ''}")
    if a.control:
        print("  These images do NOT overlap the development repositories, and")
        print("  the fold labels are random. The expected difference is zero.")
    print("\n  counterpart held out by fold:")
    print(ext.counterpart_fold.value_counts().sort_index().to_string())
    print("\n  by class:")
    print(ext.label.value_counts().to_string())
    if a.control:
        print(f"\n  Each image is scored by all {n_folds} backbones; one is "
              f"labelled 'held out' at\n  random. No backbone has any "
              f"privileged relationship to any image here.")
    else:
        print(f"\n  For each image, {n_folds - 1} backbones were fitted with "
              f"its counterpart's\n  leakage-control group present and 1 with "
              f"that group held out. Because folds\n  are group-disjoint the "
              f"unit is the group, not a single image. The external\n  image "
              f"is its own control: class, quality, preprocessing and "
              f"architecture are\n  identical across the comparison.")

    # -- per-fold predictions, kept separate ----------------------------------
    tmp = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(tmp)
    size, dim = cfg["input_size"][1], tmp.num_features
    del tmp
    tf = transforms.Compose([transforms.Resize((size, size)),
                             transforms.ToTensor(),
                             transforms.Normalize(cfg["mean"], cfg["std"])])

    names = ["cnn"] + [f"cnn_{h}" for h in a.heads]
    # correctness[model][fold] -> bool array over external images
    correct = {n: np.zeros((n_folds, len(ext)), dtype=bool) for n in names}
    conf = {n: np.zeros((n_folds, len(ext)), dtype=np.float32) for n in names}
    y = ext.label.to_numpy()

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
        p = torch.softmax(lg, 1).numpy()
        pred = np.array(labels)[p.argmax(1)]
        correct["cnn"][k] = pred == y
        conf["cnn"][k] = p.max(1)

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
                est, _ = fit_tuned(h, Xtr, tr.label.to_numpy(), cv, a.seed)
                pp = est.predict_proba(Xe)
                col = [list(est.classes_).index(l) for l in labels]
                pp = pp[:, col]
                correct[f"cnn_{h}"][k] = np.array(labels)[pp.argmax(1)] == y
                conf[f"cnn_{h}"][k] = pp.max(1)
        del model
        torch.cuda.empty_cache()

    # -- the paired contrast ---------------------------------------------------
    kstar = ext.counterpart_fold.to_numpy()
    rows = np.arange(len(ext))
    p_floor = 1.0 / a.bootstrap

    print("\n" + "=" * 78)
    if a.control:
        print("NEGATIVE CONTROL: random fold labels on non-overlapping images")
    else:
        print("Within-image paired contrast: counterpart group in training vs "
              "held out")
    print("=" * 78)
    table = []
    for n in names:
        C = correct[n]
        unseen = C[kstar, rows].astype(float)          # the one exposed to none
        mask = np.ones_like(C, dtype=bool)
        mask[kstar, rows] = False
        seen = (C * mask).sum(0) / mask.sum(0)         # mean over the other four

        d, lo, hi, p = paired_bootstrap(seen, unseen, a.bootstrap, a.seed)
        cU = conf[n][kstar, rows].mean()
        cS = (conf[n] * mask).sum(0).sum() / mask.sum()

        pstr = f"{p:.4f}" if p >= p_floor else f"<{p_floor:.4f}"
        print(f"  {n:<12} with counterpart {seen.mean():.4f}   "
              f"without {unseen.mean():.4f}   "
              f"diff {100 * d:+.2f} pp  95% CI "
              f"[{100 * lo:+.2f}, {100 * hi:+.2f}]  p={pstr}")
        print(f"  {'':<12} mean confidence: with {cS:.3f}  without {cU:.3f}")
        table.append({"model": n, "acc_with_counterpart": float(seen.mean()),
                      "acc_without_counterpart": float(unseen.mean()),
                      "diff_pp": 100 * d, "ci_lo_pp": 100 * lo,
                      "ci_hi_pp": 100 * hi, "p": p,
                      "conf_with": float(cS), "conf_without": float(cU)})

    t = pd.DataFrame(table)
    # Holm across the model family; the CNN and SVM comparisons are the
    # prespecified ones and are also reported unadjusted.
    order = np.argsort(t.p.to_numpy())
    adj, running = np.zeros(len(t)), 0.0
    for rank, i in enumerate(order):
        running = min(1.0, max(running, t.p.iloc[i] * (len(t) - rank)))
        adj[i] = running
    t["p_holm"] = adj

    print("\n" + t.round(4).to_string(index=False))
    print(f"\n  p-values are floored at 1/{a.bootstrap} = {p_floor:.4f}; report")
    print(f"  smaller values as p < {p_floor:.4f} rather than as zero.")
    print("\n  Reading this result:")
    if a.control:
        print("   - An interval spanning zero is the required outcome. It shows")
        print("     the leave-one-fold-out arithmetic does not manufacture a")
        print("     difference on its own, so the effect in the main test is")
        print("     attributable to training content rather than to the")
        print("     estimator or to variation between backbones.")
        print("   - A non-null result here invalidates the main test. Report it")
        print("     and withdraw the paired analysis rather than explaining it")
        print("     away.")
    else:
        print("   - A clear positive difference means the model performs better")
        print("     on an image when its near-identical counterpart's group was")
        print("     in that backbone's training data. The image is its own")
        print("     control, so class, quality and subpopulation cannot explain")
        print("     it. Run --control before relying on this.")
        print("   - An interval spanning zero means memorisation does not")
        print("     account for the stratified gap, which would then be better")
        print("     explained by population differences. Report that plainly.")

    out = OUT / (f"paired_control_{a.split}" if a.control
                 else f"paired_leakage_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "paired_contrast.csv", index=False)
    pd.DataFrame({"path": ext.path, "label": y,
                  "counterpart_fold": kstar,
                  "counterpart_path": ext.trained_match_path,
                  "counterpart_distance": ext.trained_distance,
                  **{f"correct_{n}_without": correct[n][kstar, rows]
                     for n in names},
                  **{f"correct_{n}_with_mean":
                     (correct[n] * (np.ones_like(correct[n], bool)))
                     .sum(0) / n_folds for n in names}}
                 ).to_csv(out / "per_image.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "run": a.run, "manifest": a.manifest, "partition": a.split,
        "mode": "negative_control" if a.control else "paired_leakage_test",
        "n_images": len(ext), "dropped": dropped, "n_folds": n_folds,
        "bootstrap": a.bootstrap, "p_floor": p_floor,
        "class_counts": ext.label.value_counts().to_dict(),
        "results": table}, indent=2, default=str))
    print(f"\n  wrote {out}/")


if __name__ == "__main__":
    main()