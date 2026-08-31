#!/usr/bin/env python3
"""
explain.py -- quantitative interpretability, error analysis, and deferral.

WHY QUANTITATIVE
----------------
Reviewer point 9 asks for interpretability analysis, not illustration. A
figure showing three well-chosen Grad-CAM overlays proves nothing: the
selection is the result. The Figshare subset ships a binary tumour mask with
every slice, so attribution can be scored rather than displayed.

METRICS
-------
  mask_energy      fraction of total saliency mass falling inside the tumour
                   mask. The primary measure.
  mask_area        fraction of the image the mask occupies. This is what
                   mask_energy would be for a uniform, uninformative map.
  concentration    mask_energy / mask_area. 1.0 means the attribution is no
                   better than uniform; higher means saliency concentrates on
                   the lesion. This ratio, not mask_energy alone, is the
                   quantity to report -- a large tumour trivially captures a
                   large share of a diffuse map.
  pointing_hit     whether the single most salient pixel lies inside the mask
                   (the standard pointing game).
  iou              overlap between the mask and the top-k% most salient pixels,
                   with k set to the mask's own area so the comparison is fair.
  background_frac  saliency mass falling outside the head altogether, using an
                   Otsu-derived head mask. High values indicate the model is
                   keying on acquisition artefacts rather than anatomy.

MASK ALIGNMENT
--------------
Only rows with source == 'figshare' are used for mask-based scoring. Some
SARTAJ rows inherited a mask_path through duplicate-cluster propagation, but
those files were independently rescaled and cropped before redistribution, so
the mask is not guaranteed to register against them. Scoring on them would
silently corrupt every number in this script.

THE 'NO TUMOUR' CLASS
---------------------
No mask exists for it, and none could. What is measured instead is where the
attribution lands: if a large share falls outside the head, the model is
reading the acquisition rather than the anatomy. Read alongside the
repository-classification probe in features.py.

Usage
-----
    python train_cnn.py --config main --epochs 20 --tag main_finetuned \\
        --save-checkpoints            # needed first, if not already saved
    python explain.py --run main_finetuned --max-per-class 150
    python explain.py --run main_finetuned --deferral
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy import ndimage
from skimage.filters import threshold_otsu
from torchvision import transforms
from tqdm import tqdm

RUNS = Path("/workspace/outputs/runs")
SPLITS = Path("/workspace/data/manifest/splits")


# =============================================================================
# Saliency scoring
# =============================================================================

def score_map(cam: np.ndarray, mask: np.ndarray | None,
              head: np.ndarray) -> dict:
    """All attribution metrics for one image."""
    cam = np.clip(cam, 0, None)
    total = cam.sum()
    if total <= 0:
        return {}

    out = {"background_frac": float(cam[~head].sum() / total)}

    if mask is None or mask.sum() == 0:
        return out

    area = mask.mean()
    energy = float(cam[mask].sum() / total)
    out.update({
        "mask_area": float(area),
        "mask_energy": energy,
        # Normalising by area is what separates "the map found the lesion"
        # from "the lesion is large".
        "concentration": float(energy / area) if area > 0 else np.nan,
        "pointing_hit": bool(mask[np.unravel_index(cam.argmax(), cam.shape)]),
    })

    # Threshold the map at its own top-(area) quantile, so the predicted region
    # has the same size as the ground-truth mask and IoU is not biased by the
    # arbitrary choice of a fixed threshold.
    k = max(1, int(round(area * cam.size)))
    thr = np.partition(cam.ravel(), -k)[-k]
    pred = cam >= thr
    inter = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    out["iou"] = float(inter / union) if union else 0.0
    return out


def head_mask(img: np.ndarray) -> np.ndarray:
    """Approximate head region by Otsu thresholding, then fill interior holes."""
    try:
        t = threshold_otsu(img)
    except Exception:
        t = img.mean()
    m = ndimage.binary_fill_holes(img > t)
    if m.sum() < 0.02 * m.size:      # degenerate threshold
        return np.ones_like(m, dtype=bool)
    return m.astype(bool)


# =============================================================================

def run_cam(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = RUNS / args.run
    meta = json.loads((run_dir / "metrics.json").read_text())
    labels = meta["labels"]
    config = meta["config"]

    df = pd.read_csv(SPLITS / f"splits_{config}_outer.csv")
    preds = pd.read_csv(run_dir / "predictions.csv")
    df = df.merge(preds[["path", "pred_cnn"]], on="path", how="left")
    df["patient_id"] = df.patient_id.fillna("")
    df["mask_path"] = df.mask_path.fillna("")

    probe = timm.create_model("efficientnet_b0", pretrained=False,
                              num_classes=len(labels))
    cfg = timm.data.resolve_model_data_config(
        timm.create_model("efficientnet_b0", pretrained=False, num_classes=0))
    size = cfg["input_size"][1]
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    del probe

    # Balanced subsample per class per fold: Grad-CAM is expensive and the
    # estimate does not improve much beyond a few hundred images per class.
    rng = np.random.default_rng(args.seed)
    picks = []
    for (fold, lab), g in df.groupby(["outer_fold", "label"]):
        n = min(len(g), max(1, args.max_per_class // df.outer_fold.nunique()))
        picks.append(g.iloc[rng.choice(len(g), n, replace=False)])
    sample = pd.concat(picks).reset_index(drop=True)
    print(f"==> scoring {len(sample)} images "
          f"({args.max_per_class} per class, spread over folds)")

    rows = []
    for fold in sorted(sample.outer_fold.unique()):
        ckpt = run_dir / f"backbone_fold{fold}.pt"
        if not ckpt.exists():
            raise SystemExit(
                f"{ckpt} not found. Re-run train_cnn.py with --save-checkpoints; "
                f"attribution must come from the model that produced the "
                f"predictions, not a retrained one.")
        model = timm.create_model("efficientnet_b0", pretrained=False,
                                  num_classes=len(labels))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval().to(device)
        cam_engine = GradCAM(model=model, target_layers=[model.conv_head])

        sub = sample[sample.outer_fold == fold]
        for _, r in tqdm(sub.iterrows(), total=len(sub), desc=f"fold {fold}"):
            img = Image.open(r.path).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            cls = labels.index(r.pred_cnn) if isinstance(r.pred_cnn, str) \
                else labels.index(r.label)
            cam = cam_engine(input_tensor=x,
                             targets=[ClassifierOutputTarget(cls)])[0]

            grey = np.asarray(img.convert("L").resize((size, size)),
                              dtype=np.float32)
            head = head_mask(grey)

            # Masks are only trustworthy on the Figshare renders themselves.
            mask = None
            if r.source == "figshare" and r.mask_path:
                mp = Path(r.mask_path)
                if mp.exists():
                    mk = np.asarray(Image.open(mp).convert("L")
                                    .resize((size, size), Image.NEAREST))
                    mask = mk > 127

            m = score_map(cam, mask, head)
            if m:
                m.update({"path": r.path, "label": r.label,
                          "pred": r.pred_cnn, "source": r.source,
                          "patient_id": r.patient_id, "outer_fold": fold,
                          "correct": r.pred_cnn == r.label,
                          "has_mask": mask is not None})
                rows.append(m)

        del model, cam_engine
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "attribution_scores.csv", index=False)
    report(res, run_dir)


def report(res: pd.DataFrame, run_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("Attribution quality, tumour classes (Figshare masks)")
    print("=" * 70)

    m = res[res.has_mask]
    if len(m):
        agg = m.groupby("label").agg(
            n=("mask_energy", "size"),
            mask_area=("mask_area", "mean"),
            mask_energy=("mask_energy", "mean"),
            concentration=("concentration", "mean"),
            pointing=("pointing_hit", "mean"),
            iou=("iou", "mean"),
        ).round(3)
        print(agg.to_string())
        print("\n  concentration = mask_energy / mask_area.")
        print("  1.0 is the uniform-map baseline: no localisation at all.")

        print("\n  correct vs incorrect predictions:")
        cc = m.groupby(["label", "correct"]).agg(
            n=("mask_energy", "size"),
            concentration=("concentration", "mean"),
            pointing=("pointing_hit", "mean"),
        ).round(3)
        print(cc.to_string())
        print("\n  If concentration is no higher on correct predictions than on")
        print("  incorrect ones, the model is not succeeding by localising the")
        print("  lesion, whatever the overlays look like.")

    print("\n" + "=" * 70)
    print("Saliency falling outside the head (all classes)")
    print("=" * 70)
    bg = res.groupby("label").background_frac.agg(["size", "mean", "median"]).round(3)
    print(bg.to_string())
    print("\n  A class with markedly more attribution outside the head is being")
    print("  decided on something other than brain anatomy.")

    if "notumor" in set(res.label):
        nt = res[res.label == "notumor"].background_frac.mean()
        others = res[res.label != "notumor"].background_frac.mean()
        print(f"\n  no tumour {nt:.3f} vs tumour classes {others:.3f} "
              f"(ratio {nt / max(others, 1e-9):.2f}x)")

    print("\n  confusion among sampled images:")
    print(pd.crosstab(res.label, res.pred).to_string())
    print(f"\n  wrote {run_dir / 'attribution_scores.csv'}")


# =============================================================================

def deferral(args) -> None:
    """
    Accuracy on retained cases as a function of how many low-confidence cases
    are referred to a radiologist.

    This is the clinically meaningful framing: a classifier that abstains on
    the cases it cannot resolve is useful at an accuracy that a classifier
    forced to answer everything is not.
    """
    run_dir = RUNS / args.run
    meta = json.loads((run_dir / "metrics.json").read_text())
    labels = meta["labels"]
    preds = pd.read_csv(run_dir / "predictions.csv")

    models = sorted({c.split("pred_", 1)[1] for c in preds.columns
                     if c.startswith("pred_")})
    y = preds.y_true.to_numpy()
    groups = preds.group_id.to_numpy()

    print("=" * 70)
    print("Deferral analysis")
    print("=" * 70)

    curves = []
    for mdl in models:
        cols = [f"prob_{mdl}_{l}" for l in labels]
        if not all(c in preds.columns for c in cols):
            continue
        P = preds[cols].to_numpy()
        conf = P.max(1)
        pred = preds[f"pred_{mdl}"].to_numpy()
        correct = (pred == y)
        order = np.argsort(conf)          # least confident first

        print(f"\n  {mdl}")
        print(f"    {'defer':>6}  {'kept':>6}  {'accuracy':>9}  "
              f"{'groups kept':>14}")
        for frac in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            n_def = int(round(frac * len(y)))
            keep = np.ones(len(y), bool)
            keep[order[:n_def]] = False
            acc = correct[keep].mean() if keep.any() else np.nan
            npat = len(np.unique(groups[keep]))
            curves.append({"model": mdl, "defer_frac": frac,
                           "n_kept": int(keep.sum()), "accuracy": float(acc),
                           "groups_kept": npat})
            print(f"    {frac:6.0%}  {keep.sum():6d}  {acc:9.4f}  {npat:14d}")

    pd.DataFrame(curves).to_csv(run_dir / "deferral_curve.csv", index=False)
    print(f"\n  wrote {run_dir / 'deferral_curve.csv'}")
    print("\n  Report the operating point, not just the curve: state the")
    print("  referral rate required to reach a target accuracy, and note that")
    print("  referred cases still consume radiologist time -- the benefit is")
    print("  triage, not replacement.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main_finetuned")
    ap.add_argument("--max-per-class", type=int, default=150)
    ap.add_argument("--deferral", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.deferral:
        deferral(a)
    else:
        run_cam(a)


if __name__ == "__main__":
    main()