#!/usr/bin/env python3
"""
xai_figure.py -- representative attribution cases, selected deterministically.

WHY SELECTION IS THE HARD PART
------------------------------
A figure of attribution maps is only evidence if the cases in it were not
chosen for how they look. The usual practice -- picking the clearest heatmaps
-- makes the figure a statement about the author's taste rather than about the
model, and a reviewer has no way to tell the difference.

The rule here is fixed in advance and stated in the caption: within each
class x correctness stratum, the case whose concentration is nearest the
median of that stratum. That is deterministic, reproducible from the stored
scores, and representative by construction rather than by assertion. No
subgroup contributes more than one case, and a subgroup containing no
misclassification contributes none: an empty panel is reported, not filled.

WHAT IS RECOMPUTED AND WHAT IS NOT
----------------------------------
Selection uses the scores already stored by occlusion_hybrid.py, so the figure
shows cases from exactly the cohort and the predictions that Table 10 reports;
no new attribution analysis is run and no new numbers enter the manuscript.
The occlusion maps themselves are not stored, only their summary scores, so
they are recomputed for the selected cases alone, with the same backbone
checkpoint and the same head refitted in the same way. Recomputation is
deterministic, so the map drawn is the map that produced the stored score.

PANELS
------
Per case: the original slice with the reference tumour contour; the occlusion
map; and the map overlaid on the slice with the contour. Beneath each, the
true and predicted labels, the model's confidence, and the three attribution
measures.

Usage
-----
    python xai_figure.py --run figshare_finetuned_v2 --head svm
    python xai_figure.py --run figshare_finetuned_v2 --select-only
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = Path("/workspace/data/manifest")
SPLITS = MANIFEST / "splits"
RUNS = Path("/workspace/outputs/runs")
FIGS = Path("/workspace/outputs/figures")

CLASSES = ["glioma", "meningioma", "pituitary"]


# =============================================================================
# Selection
# =============================================================================

def select_cases(run: str, head: str) -> pd.DataFrame:
    """One case per class x correctness, nearest the stratum median."""
    src = RUNS / run / f"occlusion_{head}.csv"
    if not src.exists():
        raise SystemExit(f"{src} not found. Run occlusion_hybrid.py first; the "
                         f"figure must come from the stored scores, not from a "
                         f"fresh analysis.")
    df = pd.read_csv(src)
    df["correct"] = df.correct.astype(bool)

    # The scores file records the image but not its mask; join it back.
    ds = pd.read_csv(MANIFEST / "dataset.csv")
    ds["mask_path"] = ds.mask_path.fillna("")
    df = df.merge(ds[["path", "mask_path"]], on="path", how="left")
    missing = df.mask_path.fillna("") == ""
    if missing.any():
        print(f"  {int(missing.sum())} scored images have no mask and cannot be "
              f"drawn with a contour; excluded from selection")
        df = df[~missing]

    rows, report = [], []
    for cls in CLASSES:
        for correct in (True, False):
            g = df[(df.label == cls) & (df.correct == correct)]
            label = "correct" if correct else "incorrect"
            report.append({"class": cls, "stratum": label, "n": len(g),
                           "median_concentration": (round(g.concentration.median(), 3)
                                                    if len(g) else None)})
            if not len(g):
                print(f"  {cls:<11} {label:<9} no cases; this stratum is "
                      f"reported as empty rather than filled")
                continue
            med = g.concentration.median()
            pick = g.assign(_d=(g.concentration - med).abs()) \
                    .sort_values(["_d", "path"]).iloc[0]
            rows.append({**pick.drop(labels=["_d"], errors="ignore").to_dict(),
                         "stratum": label,
                         "stratum_n": len(g),
                         "stratum_median_concentration": float(med)})

    sel = pd.DataFrame(rows)
    print("\n  stratum sizes and medians:")
    print(pd.DataFrame(report).to_string(index=False))
    return sel


# =============================================================================
# Rendering
# =============================================================================

def render(sel: pd.DataFrame, run: str, head: str, seed: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms

    from occlusion_hybrid import features, occlusion_map, score  # noqa: F401
    from train_eval import fit_tuned

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import json
    meta = json.loads((RUNS / run / "metrics.json").read_text())
    labels = meta["labels"]
    dev = pd.read_csv(SPLITS / f"splits_{meta['config']}_outer.csv")
    inner = pd.read_csv(SPLITS / f"splits_{meta['config']}_inner.csv")

    tmp = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(tmp)
    size, dim = cfg["input_size"][1], tmp.num_features
    del tmp
    tf = transforms.Compose([transforms.Resize((size, size)),
                             transforms.ToTensor(),
                             transforms.Normalize(cfg["mean"], cfg["std"])])

    maps: dict[str, np.ndarray] = {}
    # Group by fold so each backbone is loaded and each head refitted once.
    for fold, grp in sel.groupby("outer_fold"):
        ck = RUNS / run / f"backbone_fold{int(fold)}.pt"
        if not ck.exists():
            raise SystemExit(f"{ck} missing; rerun train_cnn.py with "
                             f"--save-checkpoints.")
        model = timm.create_model("efficientnet_b0", pretrained=False,
                                  num_classes=len(labels))
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval().to(device)

        tr = dev[dev.outer_fold != fold].reset_index(drop=True)
        sub = inner[inner.outer_fold == fold]
        tr["inner"] = tr.path.map(dict(zip(sub.path, sub.inner_fold)))
        Xtr = features(model, tr.path.tolist(), tf, device, dim, workers=4)
        asg = tr.inner.to_numpy()
        cv = [(np.where(asg != m)[0], np.where(asg == m)[0])
              for m in sorted(set(asg))]
        clf, _ = fit_tuned(head, Xtr, tr.label.to_numpy(), cv, seed)
        print(f"  fold {int(fold)}: head refitted for {len(grp)} case(s)")

        for _, r in grp.iterrows():
            x = tf(Image.open(r.path).convert("RGB"))
            ci = list(clf.classes_).index(r.pred)
            cmap, _ = occlusion_map(model, clf, x, ci, device, 32, 16)
            maps[r.path] = np.clip(cmap, 0, None)
        del model, clf
        torch.cuda.empty_cache()

    n = len(sel)
    fig, axes = plt.subplots(n, 3, figsize=(9.6, 3.4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for i, (_, r) in enumerate(sel.iterrows()):
        img = np.asarray(Image.open(r.path).convert("L").resize((size, size)),
                         dtype=np.float32)
        mask = np.asarray(Image.open(r.mask_path).convert("L")
                          .resize((size, size), Image.NEAREST)) > 127
        cmap = maps[r.path]
        cmap = cmap / cmap.max() if cmap.max() > 0 else cmap

        for j, (data, title, over) in enumerate([
                (img, "MRI + reference contour", False),
                (cmap, "Occlusion sensitivity", False),
                (img, "Overlay + contour", True)]):
            ax = axes[i, j]
            if j == 1:
                ax.imshow(data, cmap="inferno")
            else:
                ax.imshow(data, cmap="gray")
                if over:
                    ax.imshow(cmap, cmap="inferno", alpha=0.45)
            if j in (0, 2):
                ax.contour(mask, levels=[0.5], colors="#39FF6A", linewidths=1.1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=10)

        agree = "correct" if r.correct else "INCORRECT"
        axes[i, 0].set_ylabel(
            f"{r.label} — {agree}\n"
            f"predicted: {r.pred}\n"
            f"confidence {r.base_prob:.2f}\n"
            f"concentration {r.concentration:.2f}\n"
            f"pointing {int(bool(r.pointing_hit))}   IoU {r.iou:.2f}",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=68)

    fig.suptitle(
        "Representative occlusion-sensitivity cases, selected as the case "
        "nearest the median\nconcentration within each class and correctness "
        "stratum", fontsize=11, y=0.998)
    fig.tight_layout(rect=[0.09, 0.0, 1, 0.985])
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"xai_representative_cases.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}/xai_representative_cases.png and .pdf")


CAPTION = """\
Figure N. Representative occlusion-sensitivity examples for the
EfficientNetB0+SVM pipeline. Cases were selected deterministically as the
image whose attribution concentration is nearest the median of its class and
correctness stratum, from the same cohort and the same predictions reported in
Table 10; no case was chosen for its appearance. Green contours denote the
reference tumour masks. Left: the slice with the reference contour. Centre:
the occlusion map, normalised per case. Right: the map overlaid on the slice.
Correct predictions generally show more lesion-concentrated sensitivity and
incorrect predictions more diffuse or extra-lesional sensitivity. These maps
indicate where the deployed pipeline's output is sensitive to occlusion; they
do not establish clinically valid reasoning, and no radiological
interpretation of individual maps is offered.\
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="figshare_finetuned_v2")
    ap.add_argument("--head", default="svm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select-only", action="store_true",
                    help="write the case list without rendering")
    a = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    sel = select_cases(a.run, a.head)
    if not len(sel):
        raise SystemExit("no cases selected")

    cols = [c for c in ["path", "mask_path", "label", "pred", "correct",
                        "base_prob", "concentration", "pointing_hit", "iou",
                        "outer_fold", "stratum", "stratum_n",
                        "stratum_median_concentration"] if c in sel.columns]
    out_csv = FIGS / "xai_representative_cases.csv"
    sel[cols].to_csv(out_csv, index=False)
    print(f"\n  selected {len(sel)} cases -> {out_csv}\n")
    print(sel[[c for c in ["label", "stratum", "pred", "base_prob",
                           "concentration", "pointing_hit", "iou",
                           "stratum_median_concentration"]
               if c in sel.columns]].round(3).to_string(index=False))

    if not a.select_only:
        render(sel, a.run, a.head, a.seed, FIGS)

    (FIGS / "xai_caption.txt").write_text(CAPTION)
    print(f"\n  caption written to {FIGS / 'xai_caption.txt'}")


if __name__ == "__main__":
    main()