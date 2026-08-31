#!/usr/bin/env python3
"""
explain.py (v2) -- selective prediction, calibration, and quantitative
attribution.

CHANGES FROM v1
---------------
1. DEFERRAL IS NOW AT THE LEVEL OF THE INDEPENDENT UNIT, NOT THE SLICE.
   v1 ranked individual images by confidence and withheld the least confident.
   That is not a decision anyone makes: a radiologist is handed an
   examination, not a slice. Worse, with ~16 slices per glioma patient,
   image-level deferral silently withholds part of a patient while reporting
   accuracy on the rest of the same patient -- the same correlation problem
   that made image-level splitting misleading. Predictions are now pooled
   within a group, and whole groups are deferred.

2. CALIBRATION IS MEASURED, NOT ASSUMED. A deferral curve only means something
   if the scores rank uncertainty honestly. Brier score, expected and maximum
   calibration error, and a reliability table are reported for every model.
   Random forests in particular produce badly calibrated votes here
   (ECE ~0.15-0.21) and should not be presented as probabilities.

3. RISK-COVERAGE AND AURC. The deferral table is a few points on a curve; the
   area under the risk-coverage curve summarises the whole of it in one number
   and is the standard measure in the selective-prediction literature. An
   oracle bound is printed alongside so the value has a scale.

4. THE BACKGROUND-FRACTION METRIC IS REPORTED AGAINST ITS BASELINE.
   v1 reported the share of attribution falling outside the head as though a
   high value were evidence of a shortcut. It is not: the background occupies
   roughly half the resized image, and Grad-CAM for EfficientNetB0 is computed
   on a 7x7 grid and upsampled, so each cell spans ~32 pixels and inevitably
   bleeds past the skull. The measured head area is now computed and printed
   next to it. Interpret the difference, not the raw number -- and note that
   the evidence for acquisition shortcuts comes from probe_shortcut.py, which
   does not depend on this metric at all.

5. "patients kept" renamed. Only the Figshare subset carries genuine patient
   identifiers; elsewhere a group is a near-duplicate cluster standing in for
   a patient. The column follows whichever the run actually has.

Usage
-----
    python explain.py --run main_finetuned_v2 --mode deferral
    python explain.py --run main_finetuned_v2 --mode cam --max-per-class 200
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/workspace/outputs/runs")
SPLITS = Path("/workspace/data/manifest/splits")


# =============================================================================
# Calibration
# =============================================================================

def calibration_table(conf: np.ndarray, correct: np.ndarray,
                      n_bins: int = 15) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                         "mean_confidence": float(conf[m].mean()),
                         "accuracy": float(correct[m].mean()),
                         "gap": float(correct[m].mean() - conf[m].mean())})
    return pd.DataFrame(rows)


def calibration_metrics(conf, correct, P=None, y_idx=None, k=None) -> dict:
    t = calibration_table(conf, correct)
    w = t.n / t.n.sum()
    out = {
        "ece": float((w * t.gap.abs()).sum()),
        "mce": float(t.gap.abs().max()),
        "mean_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
        # Positive means the model is under-confident, negative over-confident.
        "confidence_gap": float(correct.mean() - conf.mean()),
    }
    if P is not None:
        out["brier"] = float(((P - np.eye(k)[y_idx]) ** 2).sum(1).mean())
    return out


# =============================================================================
# Selective prediction
# =============================================================================

def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple:
    """
    Risk (error rate) as a function of coverage, retaining the most confident
    cases first. AURC summarises the curve; the oracle bound -- perfect ranking
    of errors to the bottom -- gives it a scale.
    """
    order = np.argsort(-conf)
    err = (~correct.astype(bool)).astype(float)[order]
    n = len(err)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)

    oracle = np.sort(err)                       # all correct cases first
    orisk = np.cumsum(oracle) / np.arange(1, n + 1)
    return cov, risk, float(risk.mean()), float(orisk.mean())


def group_level(df: pd.DataFrame, labels: list[str], model: str):
    """
    Pool slice probabilities within a group and produce one prediction per
    independent unit. Averaging probabilities rather than taking a majority
    vote keeps the confidence continuous, which the deferral ranking needs.
    """
    cols = [f"prob_{model}_{l}" for l in labels]
    g = df.groupby("group_id")
    P = g[cols].mean().to_numpy()
    truth = g.y_true.first().to_numpy()
    size = g.size().to_numpy()
    ids = np.array(list(g.groups.keys()))
    pred = np.array(labels)[P.argmax(1)]
    return ids, truth, pred, P, size


def deferral(args) -> None:
    run_dir = RUNS / args.run
    meta = json.loads((run_dir / "metrics.json").read_text())
    labels = meta["labels"]
    unit = meta.get("bootstrap_unit", "group")
    preds = pd.read_csv(run_dir / "predictions.csv")

    models = sorted({c.split("pred_", 1)[1] for c in preds.columns
                     if c.startswith("pred_")})
    y = preds.y_true.to_numpy()

    print("=" * 74)
    print(f"Selective prediction -- {args.run}")
    print(f"independent unit: {unit}   "
          f"({preds.group_id.nunique()} units, {len(preds)} images)")
    print("=" * 74)

    cal_rows, curve_rows, rel_rows = [], [], []

    for mdl in models:
        cols = [f"prob_{mdl}_{l}" for l in labels]
        if not all(c in preds.columns for c in cols):
            continue

        # -- calibration, at both levels --------------------------------------
        Pi = preds[cols].to_numpy()
        ci = Pi.max(1)
        corr_i = (preds[f"pred_{mdl}"].to_numpy() == y)
        yi = np.array([labels.index(v) for v in y])
        m_img = calibration_metrics(ci, corr_i.astype(float), Pi, yi, len(labels))

        ids, gt, gp, Pg, gsize = group_level(preds, labels, mdl)
        cg = Pg.max(1)
        corr_g = (gp == gt)
        yg = np.array([labels.index(v) for v in gt])
        m_grp = calibration_metrics(cg, corr_g.astype(float), Pg, yg, len(labels))

        for lvl, m in [("image", m_img), (unit, m_grp)]:
            cal_rows.append({"model": mdl, "level": lvl, **m})

        for lo, hi, n, mc, acc, gap in calibration_table(cg, corr_g.astype(float)).values:
            rel_rows.append({"model": mdl, "bin_lo": lo, "bin_hi": hi,
                             "n": n, "mean_confidence": mc, "accuracy": acc})

        # -- risk-coverage ----------------------------------------------------
        _, _, aurc, oracle = risk_coverage(cg, corr_g)

        print(f"\n  {mdl}")
        print(f"    calibration  image: ECE {m_img['ece']:.4f}  "
              f"Brier {m_img['brier']:.4f}   |   "
              f"{unit}: ECE {m_grp['ece']:.4f}  Brier {m_grp['brier']:.4f}")
        print(f"    AURC {aurc:.4f}  (oracle {oracle:.4f})")
        if m_grp["ece"] > 0.10:
            print(f"    !! ECE {m_grp['ece']:.3f}: these scores are not usable "
                  f"as probabilities.\n       Exclude this model from the "
                  f"deferral claim or recalibrate it.")

        # -- deferral table, whole units withheld -----------------------------
        order = np.argsort(cg)                  # least confident first
        print(f"    {'defer':>6}  {'units':>6}  {'images':>7}  "
              f"{'unit acc':>9}  {'image acc':>10}")
        for frac in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            n_def = int(round(frac * len(ids)))
            keep = np.ones(len(ids), bool)
            keep[order[:n_def]] = False
            kept_ids = set(ids[keep])
            im = preds.group_id.isin(kept_ids).to_numpy()
            uacc = corr_g[keep].mean() if keep.any() else np.nan
            iacc = corr_i[im].mean() if im.any() else np.nan
            curve_rows.append({"model": mdl, "defer_frac": frac,
                               "units_kept": int(keep.sum()),
                               "images_kept": int(im.sum()),
                               "unit_accuracy": float(uacc),
                               "image_accuracy": float(iacc)})
            print(f"    {frac:6.0%}  {keep.sum():6d}  {im.sum():7d}  "
                  f"{uacc:9.4f}  {iacc:10.4f}")

    pd.DataFrame(cal_rows).to_csv(run_dir / "calibration.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(run_dir / "deferral_curve.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(run_dir / "reliability_bins.csv", index=False)

    print(f"\n  wrote calibration.csv, deferral_curve.csv, reliability_bins.csv")
    print("\n  Reporting notes:")
    print("    - Deferral withholds whole units, so a referred case costs a")
    print("      full examination of radiologist time, not one slice. Report")
    print("      the images column too: it is the real workload transferred.")
    print("    - Frame this as triage or selective prediction, never as")
    print("      autonomous diagnosis.")
    print("    - Accuracy on retained cases is conditional on the model's own")
    print("      confidence ordering and is not a diagnostic accuracy that")
    print("      would transfer to an unselected population.")


# =============================================================================
# Attribution
# =============================================================================

def cam(args) -> None:
    import timm
    import torch
    from PIL import Image
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from torchvision import transforms
    from tqdm import tqdm

    def head_mask(g):
        try:
            t = threshold_otsu(g)
        except Exception:
            t = g.mean()
        m = ndimage.binary_fill_holes(g > t)
        return m.astype(bool) if m.sum() >= 0.02 * m.size else np.ones_like(m, bool)

    def score(cmap, mask, head):
        cmap = np.clip(cmap, 0, None)
        tot = cmap.sum()
        if tot <= 0:
            return {}
        out = {"background_frac": float(cmap[~head].sum() / tot),
               "head_area": float(head.mean())}
        if mask is None or mask.sum() == 0:
            return out
        area = mask.mean()
        energy = float(cmap[mask].sum() / tot)
        out.update({"mask_area": float(area), "mask_energy": energy,
                    "concentration": float(energy / area) if area else np.nan,
                    "pointing_hit": bool(mask[np.unravel_index(cmap.argmax(),
                                                               cmap.shape)])})
        k = max(1, int(round(area * cmap.size)))
        thr = np.partition(cmap.ravel(), -k)[-k]
        p = cmap >= thr
        u = np.logical_or(p, mask).sum()
        out["iou"] = float(np.logical_and(p, mask).sum() / u) if u else 0.0
        return out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = RUNS / args.run
    meta = json.loads((run_dir / "metrics.json").read_text())
    labels = meta["labels"]
    df = pd.read_csv(SPLITS / f"splits_{meta['config']}_outer.csv")
    df = df.merge(pd.read_csv(run_dir / "predictions.csv")[["path", "pred_cnn"]],
                  on="path", how="left")
    df["mask_path"] = df.mask_path.fillna("")

    tmp = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(tmp)
    size = cfg["input_size"][1]
    del tmp
    tf = transforms.Compose([transforms.Resize((size, size)),
                             transforms.ToTensor(),
                             transforms.Normalize(cfg["mean"], cfg["std"])])

    rng = np.random.default_rng(args.seed)
    per_fold = max(1, args.max_per_class // df.outer_fold.nunique())
    sample = pd.concat([g.iloc[rng.choice(len(g), min(len(g), per_fold),
                                          replace=False)]
                        for _, g in df.groupby(["outer_fold", "label"])
                        ]).reset_index(drop=True)
    print(f"==> scoring {len(sample)} images")

    rows = []
    for fold in sorted(sample.outer_fold.unique()):
        ck = run_dir / f"backbone_fold{fold}.pt"
        if not ck.exists():
            raise SystemExit(f"{ck} missing. Re-run train_cnn.py with "
                             f"--save-checkpoints: attribution must come from "
                             f"the model that made the predictions.")
        model = timm.create_model("efficientnet_b0", pretrained=False,
                                  num_classes=len(labels))
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval().to(device)
        engine = GradCAM(model=model, target_layers=[model.conv_head])

        sub = sample[sample.outer_fold == fold]
        for _, r in tqdm(sub.iterrows(), total=len(sub), desc=f"fold {fold}"):
            img = Image.open(r.path).convert("RGB")
            cls = labels.index(r.pred_cnn if isinstance(r.pred_cnn, str) else r.label)
            cm = engine(input_tensor=tf(img).unsqueeze(0).to(device),
                        targets=[ClassifierOutputTarget(cls)])[0]
            grey = np.asarray(img.convert("L").resize((size, size)), np.float32)
            hm = head_mask(grey)
            # Masks register only against the Figshare renders themselves;
            # SARTAJ copies were rescaled and cropped before redistribution.
            mk = None
            if r.source == "figshare" and r.mask_path and Path(r.mask_path).exists():
                mk = np.asarray(Image.open(r.mask_path).convert("L")
                                .resize((size, size), Image.NEAREST)) > 127
            s = score(cm, mk, hm)
            if s:
                s.update({"path": r.path, "label": r.label, "pred": r.pred_cnn,
                          "source": r.source, "outer_fold": fold,
                          "correct": r.pred_cnn == r.label,
                          "has_mask": mk is not None})
                rows.append(s)
        del model, engine
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "attribution_scores.csv", index=False)

    print("\n" + "=" * 74)
    print("Attribution concentration (Figshare masks only)")
    print("=" * 74)
    m = res[res.has_mask]
    if len(m):
        print(m.groupby("label").agg(
            n=("mask_energy", "size"), mask_area=("mask_area", "mean"),
            mask_energy=("mask_energy", "mean"),
            concentration=("concentration", "mean"),
            pointing=("pointing_hit", "mean"), iou=("iou", "mean")
        ).round(3).to_string())
        print("\n  concentration = mask_energy / mask_area; 1.0 is the "
              "uniform-map baseline.")
        print("\n  by correctness:")
        print(m.groupby(["label", "correct"]).agg(
            n=("mask_energy", "size"),
            concentration=("concentration", "mean"),
            pointing=("pointing_hit", "mean")).round(3).to_string())
        print("\n  Concentration higher on correct than incorrect predictions "
              "means the model\n  succeeds by attending to the lesion. Where "
              "the relation inverts, it does not.")

    print("\n" + "=" * 74)
    print("Attribution outside the head -- REPORT AGAINST THE BASELINE")
    print("=" * 74)
    bg = res.groupby("label").agg(n=("background_frac", "size"),
                                  outside=("background_frac", "mean"),
                                  head_area=("head_area", "mean")).round(3)
    bg["background_area"] = (1 - bg.head_area).round(3)
    bg["excess"] = (bg.outside - bg.background_area).round(3)
    print(bg.to_string())
    print("\n  'excess' is what matters: attribution outside the head minus the")
    print("  share of the image the background actually occupies. Values near")
    print("  zero mean the map is diffuse at the 7x7 Grad-CAM resolution, not")
    print("  that the model attends to the background. Do not present the raw")
    print("  'outside' column as evidence of a shortcut -- the evidence for that")
    print("  comes from probe_shortcut.py and does not rest on this metric.")

    print("\n  confusion among sampled images:")
    print(pd.crosstab(res.label, res.pred).to_string())
    print(f"\n  wrote {run_dir / 'attribution_scores.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main_finetuned_v2")
    ap.add_argument("--mode", choices=["deferral", "cam"], default="deferral")
    ap.add_argument("--max-per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    (deferral if a.mode == "deferral" else cam)(a)


if __name__ == "__main__":
    main()