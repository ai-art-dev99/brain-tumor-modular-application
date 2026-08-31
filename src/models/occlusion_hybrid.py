#!/usr/bin/env python3
"""
occlusion_hybrid.py -- attribution for the whole CNN + classical-head pipeline.

WHY THIS EXISTS
---------------
Grad-CAM differentiates through the CNN's own classification head. In the
hybrid models that head is discarded: the prediction comes from an SVM or a
logistic regression fitted on the pooled 1,280-d features. A Grad-CAM map
therefore explains the shared feature extractor, not the decision that was
actually reported, and saying otherwise would be a misstatement.

Occlusion sensitivity is model-agnostic and runs on the deployed pipeline:

    image -> occlude a patch -> EfficientNetB0 -> 1,280-d -> head -> score

The attribution value of a patch is the drop in the head's score for the
predicted class when that patch is masked. No gradients are needed, so the
head can be anything.

The same mask-based metrics as explain.py are computed, so the two are
directly comparable: concentration relative to the tumour mask area,
the pointing game, and area-matched IoU.

COST
----
A 224x224 image with a 32-pixel patch at stride 16 gives 169 occluded copies.
Batched, that is a fraction of a second per image on a GPU. Budget a few
minutes for a few hundred images per fold.

Usage
-----
    python occlusion_hybrid.py --run main_finetuned_v2 --head svm \\
        --max-per-class 100
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train_eval import fit_tuned

SPLITS = Path("/workspace/data/manifest/splits")
RUNS = Path("/workspace/outputs/runs")


class Plain(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), i


@torch.no_grad()
def features(model, paths, tf, device, dim, bs=64, workers=8):
    dl = DataLoader(Plain(paths, tf), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=True)
    out = np.empty((len(paths), dim), dtype=np.float32)
    for xb, idx in dl:
        with torch.autocast("cuda", enabled=device == "cuda"):
            z = model.forward_features(xb.to(device, non_blocking=True))
            z = model.forward_head(z, pre_logits=True)
        out[idx.numpy()] = z.float().cpu().numpy()
    return out


@torch.no_grad()
def occlusion_map(model, head, x, cls_idx, device, patch, stride, chunk=64):
    """
    Score drop per occluded patch, for the deployed pipeline.

    The occluding value is zero in normalised space, i.e. the dataset mean
    rather than black. Using true black would itself be an out-of-distribution
    intensity and would inflate every patch's apparent importance.
    """
    _, H, W = x.shape
    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    coords = [(a, b) for a in ys for b in xs]

    base_f = model.forward_head(
        model.forward_features(x.unsqueeze(0).to(device)), pre_logits=True)
    base = head.predict_proba(base_f.float().cpu().numpy())[0][cls_idx]

    drops = np.zeros(len(coords), dtype=np.float32)
    for s in range(0, len(coords), chunk):
        batch = coords[s:s + chunk]
        xb = x.unsqueeze(0).repeat(len(batch), 1, 1, 1).clone()
        for j, (a, b) in enumerate(batch):
            xb[j, :, a:a + patch, b:b + patch] = 0.0
        with torch.autocast("cuda", enabled=device == "cuda"):
            f = model.forward_head(
                model.forward_features(xb.to(device)), pre_logits=True)
        p = head.predict_proba(f.float().cpu().numpy())[:, cls_idx]
        drops[s:s + len(batch)] = base - p

    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    for (a, b), d in zip(coords, drops):
        acc[a:a + patch, b:b + patch] += d
        cnt[a:a + patch, b:b + patch] += 1
    return np.divide(acc, np.maximum(cnt, 1)), float(base)


def head_mask(g):
    try:
        t = threshold_otsu(g)
    except Exception:
        t = g.mean()
    m = ndimage.binary_fill_holes(g > t)
    return m.astype(bool) if m.sum() >= 0.02 * m.size else np.ones_like(m, bool)


def score(cmap, mask):
    """Only positive drops count: a patch whose removal raises the score is
    evidence against the class, not attribution for it."""
    cmap = np.clip(cmap, 0, None)
    tot = cmap.sum()
    if tot <= 0 or mask is None or mask.sum() == 0:
        return {}
    area = float(mask.mean())
    energy = float(cmap[mask].sum() / tot)
    k = max(1, int(round(area * cmap.size)))
    thr = np.partition(cmap.ravel(), -k)[-k]
    p = cmap >= thr
    u = np.logical_or(p, mask).sum()
    return {"mask_area": area, "mask_energy": energy,
            "concentration": energy / area if area else np.nan,
            "pointing_hit": bool(mask[np.unravel_index(cmap.argmax(), cmap.shape)]),
            "iou": float(np.logical_and(p, mask).sum() / u) if u else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="main_finetuned_v2")
    ap.add_argument("--head", default="svm")
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = RUNS / a.run
    meta = json.loads((run_dir / "metrics.json").read_text())
    labels = meta["labels"]
    df = pd.read_csv(SPLITS / f"splits_{meta['config']}_outer.csv")
    inner = pd.read_csv(SPLITS / f"splits_{meta['config']}_inner.csv")
    df["mask_path"] = df.mask_path.fillna("")

    tmp = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(tmp)
    size, dim = cfg["input_size"][1], tmp.num_features
    del tmp
    tf = transforms.Compose([transforms.Resize((size, size)),
                             transforms.ToTensor(),
                             transforms.Normalize(cfg["mean"], cfg["std"])])

    rng = np.random.default_rng(a.seed)
    per_fold = max(1, a.max_per_class // df.outer_fold.nunique())
    # Masks register only against the Figshare renders; SARTAJ copies were
    # rescaled before redistribution, so scoring them would be meaningless.
    pool = df[(df.source == "figshare") & (df.mask_path != "")]
    sample = pd.concat([g.iloc[rng.choice(len(g), min(len(g), per_fold),
                                          replace=False)]
                        for _, g in pool.groupby(["outer_fold", "label"])
                        ]).reset_index(drop=True)
    print(f"==> {a.run}, head '{a.head}': occluding {len(sample)} images "
          f"(patch {a.patch}, stride {a.stride})")

    rows = []
    for fold in sorted(sample.outer_fold.unique()):
        ck = run_dir / f"backbone_fold{fold}.pt"
        if not ck.exists():
            raise SystemExit(f"{ck} missing; rerun train_cnn.py with "
                             f"--save-checkpoints.")
        model = timm.create_model("efficientnet_b0", pretrained=False,
                                  num_classes=len(labels))
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval().to(device)

        # Refit the head exactly as train_cnn did for this fold, so the
        # explanation targets the reported decision rather than a new model.
        tr = df[df.outer_fold != fold].reset_index(drop=True)
        sub = inner[inner.outer_fold == fold]
        tr["inner"] = tr.path.map(dict(zip(sub.path, sub.inner_fold)))
        Xtr = features(model, tr.path.tolist(), tf, device, dim,
                       workers=a.workers)
        asg = tr.inner.to_numpy()
        cv = [(np.where(asg != m)[0], np.where(asg == m)[0])
              for m in sorted(set(asg))]
        head, params = fit_tuned(a.head, Xtr, tr.label.to_numpy(), cv, a.seed)
        print(f"  fold {fold}: head refitted {params}")

        sf = sample[sample.outer_fold == fold]
        for _, r in tqdm(sf.iterrows(), total=len(sf), desc=f"fold {fold}"):
            img = Image.open(r.path).convert("RGB")
            x = tf(img)
            f0 = features(model, [r.path], tf, device, dim, workers=0)
            pred = head.predict(f0)[0]
            ci = list(head.classes_).index(pred)
            cmap, base = occlusion_map(model, head, x, ci, device,
                                       a.patch, a.stride)
            mk = np.asarray(Image.open(r.mask_path).convert("L")
                            .resize((size, size), Image.NEAREST)) > 127
            s = score(cmap, mk)
            if s:
                s.update({"path": r.path, "label": r.label, "pred": pred,
                          "correct": pred == r.label, "base_prob": base,
                          "outer_fold": fold})
                rows.append(s)
        del model, head
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    out = run_dir / f"occlusion_{a.head}.csv"
    res.to_csv(out, index=False)

    print("\n" + "=" * 70)
    print(f"Occlusion attribution for the full pipeline (head: {a.head})")
    print("=" * 70)
    print(res.groupby("label").agg(
        n=("mask_energy", "size"), mask_area=("mask_area", "mean"),
        mask_energy=("mask_energy", "mean"),
        concentration=("concentration", "mean"),
        pointing=("pointing_hit", "mean"), iou=("iou", "mean")
    ).round(3).to_string())
    print("\n  by correctness:")
    print(res.groupby(["label", "correct"]).agg(
        n=("mask_energy", "size"), concentration=("concentration", "mean"),
        pointing=("pointing_hit", "mean")).round(3).to_string())
    print("\n  concentration = mask_energy / mask_area; 1.0 is the uniform "
          "baseline.")
    print("  Compare against the Grad-CAM figures in attribution_scores.csv:")
    print("  agreement between a gradient method on the backbone and an")
    print("  occlusion method on the deployed pipeline is reassuring;")
    print("  disagreement means the Grad-CAM maps were describing the CNN head")
    print("  rather than the hybrid decision, and only these numbers should be")
    print("  reported for the hybrid.")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()