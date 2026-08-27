#!/usr/bin/env python3
"""
features.py -- extract EfficientNetB0 features, and probe for an acquisition
shortcut.

TWO FIXES TO THE ORIGINAL PIPELINE
----------------------------------
1. FEATURE DIMENSION. The original code removed the classifier head from the
   NVIDIA torch.hub EfficientNet. In that implementation the pooling layer sits
   *inside* the classifier, so removing it also removed the pooling: the
   "global average pooled" vector was in fact a flattened 10x10x1280 feature
   map of 128,000 dimensions. timm's num_classes=0, global_pool='avg' returns
   the intended 1280-d vector. This is a ~100x reduction in the input
   dimensionality of every downstream classifier, and it is the reason the SVM
   previously needed ~400 s to fit.

2. INPUT RESOLUTION AND NORMALISATION. The original resized to 299x299 and
   described this as matching EfficientNetB0; the model's native resolution is
   224x224 (299 belongs to InceptionV3). Rather than hard-coding either, the
   transform is now resolved from the model's own published data config.

NO AUGMENTATION IS APPLIED HERE. The original pipeline passed the same
transform -- including ColorJitter, RandomHorizontalFlip and RandomRotation --
to the evaluation loader, so every evaluation and every extracted feature was
a random draw and could not be reproduced. Extraction is deterministic.

MODES
-----
  frozen    ImageNet weights, no training. One pass over the dataset. Suitable
            for the shortcut probe and for a transfer-learning baseline.
  finetune  Not implemented here; see train_eval.py. Fine-tuning must happen
            once per outer fold, on that fold's training portion only, or the
            extracted features carry information about the test fold.

THE SHORTCUT PROBE
------------------
In this benchmark the 'no tumour' class is drawn entirely from BR35H while the
tumour classes are drawn almost entirely from the Figshare collection. The two
repositories differ in MRI sequence: Figshare is T1-weighted contrast-enhanced
throughout, while BR35H visibly mixes T2 and FLAIR. A model can therefore
score highly on 'no tumour' by recognising the acquisition, never having
learned anything about pathology.

The probe trains a linear classifier to predict the SOURCE REPOSITORY from the
same features used for diagnosis, under the same grouped splits. If the
repositories are near-perfectly separable, the 'no tumour' label is recoverable
from acquisition signature alone, and the reported per-class accuracy for that
class cannot be interpreted as diagnostic performance.

Usage
-----
    python features.py --config main --mode frozen
    python features.py --config main --probe
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

MANIFEST = Path("/workspace/data/manifest")
SPLITS = MANIFEST / "splits"
FEATURES = Path("/workspace/data/features")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageList(Dataset):
    """Paths in, transformed tensors out. Greyscale MRI is replicated to RGB
    because the ImageNet-pretrained stem expects three channels."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), i


def build_backbone(device: str):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=0,      # drop the classifier
        global_pool="avg",  # keep the pooling -> 1280-d
    )
    model.eval().to(device)

    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)

    print("  backbone      : efficientnet_b0 (timm, ImageNet weights)")
    print(f"  input size    : {cfg['input_size']}")
    print(f"  interpolation : {cfg['interpolation']}, crop {cfg['crop_pct']}")
    print(f"  normalisation : mean {cfg['mean']}, std {cfg['std']}")
    print(f"  feature dim   : {model.num_features}")
    return model, tf, cfg


@torch.no_grad()
def extract(model, tf, paths, device, batch_size=64, workers=8) -> np.ndarray:
    ds = ImageList(paths, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)
    out = np.empty((len(ds), model.num_features), dtype=np.float32)
    for xb, idx in tqdm(dl, unit="batch"):
        f = model(xb.to(device, non_blocking=True))
        out[idx.numpy()] = f.float().cpu().numpy()
    return out


def run_extract(config: str, seed: int, batch_size: int, workers: int) -> None:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FEATURES.mkdir(parents=True, exist_ok=True)

    split_file = SPLITS / f"splits_{config}_outer.csv"
    if not split_file.exists():
        raise SystemExit(f"{split_file} not found. Run split.py first.")
    df = pd.read_csv(split_file)
    print(f"==> {config}: {len(df)} images")

    model, tf, cfg = build_backbone(device)
    feats = extract(model, tf, df.path.tolist(), device, batch_size, workers)

    np.save(FEATURES / f"{config}_frozen.npy", feats)
    df.to_csv(FEATURES / f"{config}_index.csv", index=False)
    print(f"\n  features {feats.shape} -> "
          f"{FEATURES / f'{config}_frozen.npy'}")
    print(f"  L2 norm: mean {np.linalg.norm(feats, axis=1).mean():.2f}, "
          f"zero-variance dims: {int((feats.std(0) == 0).sum())}")


def run_probe(config: str, seed: int) -> None:
    """Can a linear model recover the repository from the diagnostic features?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.load(FEATURES / f"{config}_frozen.npy")
    df = pd.read_csv(FEATURES / f"{config}_index.csv")

    print("=" * 70)
    print("Acquisition-shortcut probe")
    print("=" * 70)
    print("\n  class composition by repository:")
    print(pd.crosstab(df.source, df.label).to_string())

    # Any class drawn from exactly one repository is, in principle, decidable
    # from acquisition signature alone.
    per_class_src = df.groupby("label").source.nunique()
    single = per_class_src[per_class_src == 1]
    if len(single):
        print(f"\n  classes drawn from a single repository: "
              f"{', '.join(single.index)}")

    y_src = df.source.to_numpy()
    groups = df.group_id.to_numpy()
    folds = df.outer_fold.to_numpy()

    preds = np.empty(len(df), dtype=object)
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
        clf.fit(X[tr], y_src[tr])
        preds[te] = clf.predict(X[te])

    acc = accuracy_score(y_src, preds)
    labs = sorted(set(y_src))
    cm = confusion_matrix(y_src, preds, labels=labs)

    print(f"\n  repository classification accuracy: {acc:.4f}")
    print("  (grouped 5-fold, same splits as the diagnostic task)")
    print("\n  confusion matrix (rows true, cols predicted):")
    print(pd.DataFrame(cm, index=labs, columns=labs).to_string())

    print()
    if acc > 0.97:
        print("  The repositories are near-perfectly separable from the very")
        print("  features used for diagnosis. Because 'no tumour' is sourced")
        print("  exclusively from BR35H, that class is recoverable without any")
        print("  reference to pathology. Per-class figures for 'no tumour' on")
        print("  this benchmark therefore measure acquisition recognition at")
        print("  least as much as diagnosis, and the ~99% routinely reported")
        print("  for it in the literature should be read in that light.")
    elif acc > 0.85:
        print("  Substantial repository signal is present. Report it and treat")
        print("  the single-source classes with caution.")
    else:
        print("  Repository signal is weak; the shortcut concern is not")
        print("  supported by this probe.")

    pd.DataFrame({"path": df.path, "true_source": y_src,
                  "pred_source": preds}).to_csv(
        FEATURES / f"{config}_source_probe.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="main")
    ap.add_argument("--mode", choices=["frozen"], default="frozen")
    ap.add_argument("--probe", action="store_true",
                    help="run the acquisition-shortcut probe on existing features")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    if a.probe:
        run_probe(a.config, a.seed)
    else:
        run_extract(a.config, a.seed, a.batch_size, a.workers)


if __name__ == "__main__":
    main()