#!/usr/bin/env python3
"""
probe_shortcut.py -- does the model separate repositories by acquisition
signature, or merely by tumour phenotype?

THE OBJECTION THIS ANSWERS
--------------------------
A probe that recovers the source repository at 98.7% accuracy is suggestive
but not conclusive, because diagnosis and source are confounded in this
benchmark: glioma comes only from Figshare, 'no tumour' only from BR35H. A
reviewer can reasonably reply that the probe is simply recognising the
pathology, which happens to be source-specific.

Three variants separate the two explanations.

  full        whole image. The original probe. Confounded.

  background  the head is masked out and only what surrounds it remains:
              framing, padding, noise floor, compression artefacts, field of
              view. No anatomy and therefore no pathology is present. If the
              repository is still recoverable here, the signature is in the
              acquisition and storage pipeline, and the confound explanation
              fails. This is the decisive test.

  brain       the complement: background masked out, optionally cropped to the
              head bounding box and rescaled, which also removes framing and
              scale cues. Performance here is what a model would have to rely
              on if the shortcut were unavailable.

CLASS-CONDITIONAL PROBE
-----------------------
--class-conditional holds the diagnosis fixed and asks whether the source is
still recoverable. Only meningioma supports this: 702 Figshare against 133
SARTAJ images survive deduplication. Pituitary has just 35 SARTAJ survivors,
too few to interpret, and BR35H shares no class with the others at all. The
background probe is therefore the stronger evidence; this one corroborates.

READING THE NUMBERS
-------------------
Sources are heavily imbalanced (3,038 Figshare against 607 BR35H and 168
SARTAJ), so plain accuracy has a high floor. Balanced accuracy and per-source
recall are reported alongside, and a majority-class baseline is printed for
reference. Splits are the same group-disjoint folds used for diagnosis, so no
patient contributes to both sides.

Usage
-----
    python probe_shortcut.py --variant full
    python probe_shortcut.py --variant background
    python probe_shortcut.py --variant brain --crop
    python probe_shortcut.py --variant background --pair figshare br35h
    python probe_shortcut.py --variant background --class-conditional meningioma
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

SPLITS = Path("/workspace/data/manifest/splits")
OUT = Path("/workspace/outputs/probes")


def head_mask(grey: np.ndarray) -> np.ndarray:
    """Otsu threshold, holes filled, largest connected component kept."""
    try:
        t = threshold_otsu(grey)
    except Exception:
        t = grey.mean()
    m = ndimage.binary_fill_holes(grey > t)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    if m.sum() < 0.02 * m.size:      # degenerate threshold
        m = np.ones_like(m, dtype=bool)
    return m.astype(bool)


class MaskedImages(Dataset):
    """Applies the head mask before the network transform."""

    def __init__(self, paths, variant, crop, size, mean, std):
        self.paths, self.variant, self.crop = list(paths), variant, crop
        self.size = size
        self.norm = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L").resize(
            (self.size, self.size))
        a = np.asarray(img, dtype=np.uint8)

        if self.variant != "full":
            m = head_mask(a.astype(np.float32))
            if self.variant == "brain":
                a = np.where(m, a, 0).astype(np.uint8)
                if self.crop and m.any():
                    # Cropping to the head bounding box and rescaling removes
                    # framing and field-of-view cues as well as the background
                    # itself, which the plain mask leaves intact.
                    ys, xs = np.where(m)
                    a = np.asarray(
                        Image.fromarray(a[ys.min():ys.max() + 1,
                                          xs.min():xs.max() + 1])
                        .resize((self.size, self.size)), dtype=np.uint8)
            else:                                   # background
                a = np.where(m, 0, a).astype(np.uint8)

        rgb = np.repeat(a[:, :, None], 3, axis=2)
        return self.norm(Image.fromarray(rgb)), i


@torch.no_grad()
def features_for(paths, variant, crop, batch_size, workers):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("efficientnet_b0", pretrained=True,
                              num_classes=0, global_pool="avg").eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    ds = MaskedImages(paths, variant, crop, cfg["input_size"][1],
                      cfg["mean"], cfg["std"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)
    out = np.empty((len(ds), model.num_features), dtype=np.float32)
    for xb, idx in tqdm(dl, unit="batch", desc=f"{variant}{'+crop' if crop else ''}"):
        out[idx.numpy()] = model(xb.to(device)).float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out


def probe(X, y_src, folds, seed) -> dict:
    """Grouped out-of-fold source classification, using the diagnostic splits."""
    preds = np.empty(len(y_src), dtype=object)
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        if len(set(y_src[tr])) < 2:
            preds[te] = y_src[tr][0]
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=seed))
        clf.fit(X[tr], y_src[tr])
        preds[te] = clf.predict(X[te])
    labs = sorted(set(y_src))
    return {"pred": preds, "labels": labs,
            "accuracy": float(accuracy_score(y_src, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_src, preds)),
            "confusion": confusion_matrix(y_src, preds, labels=labs).tolist(),
            "report": classification_report(y_src, preds, labels=labs,
                                            zero_division=0, output_dict=True)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="main")
    ap.add_argument("--variant", choices=["full", "brain", "background"],
                    default="background")
    ap.add_argument("--crop", action="store_true",
                    help="brain variant only: crop to head bbox and rescale")
    ap.add_argument("--pair", nargs=2, default=None,
                    metavar=("SRC_A", "SRC_B"),
                    help="restrict to two repositories, e.g. figshare br35h")
    ap.add_argument("--class-conditional", default=None,
                    help="hold diagnosis fixed, e.g. meningioma")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SPLITS / f"splits_{a.config}_outer.csv")

    if a.class_conditional:
        df = df[df.label == a.class_conditional]
        print(f"==> class-conditional on '{a.class_conditional}'")
    if a.pair:
        df = df[df.source.isin(a.pair)]
        print(f"==> restricted to {a.pair[0]} vs {a.pair[1]}")
    df = df.reset_index(drop=True)

    counts = df.source.value_counts()
    print(f"\n  {len(df)} images")
    print(counts.to_string())
    if len(counts) < 2:
        raise SystemExit("fewer than two sources remain; nothing to probe.")
    if counts.min() < 30:
        print(f"\n  WARNING: smallest source has {counts.min()} images. "
              f"Treat this probe as indicative only;\n  with so few examples "
              f"the estimate is unstable regardless of what it shows.")

    print(f"\n  majority-class baseline accuracy: {counts.max() / len(df):.4f}")
    print(f"  chance balanced accuracy:         {1 / len(counts):.4f}")

    X = features_for(df.path, a.variant, a.crop, a.batch_size, a.workers)
    r = probe(X, df.source.to_numpy(), df.outer_fold.to_numpy(), a.seed)

    print(f"\n  variant: {a.variant}{' + crop' if a.crop else ''}")
    print(f"  accuracy          {r['accuracy']:.4f}")
    print(f"  balanced accuracy {r['balanced_accuracy']:.4f}")
    print("\n  confusion (rows true, cols predicted):")
    print(pd.DataFrame(r["confusion"], index=r["labels"],
                       columns=r["labels"]).to_string())
    print("\n  per-source recall:")
    for l in r["labels"]:
        print(f"    {l:<10} {r['report'][l]['recall']:.4f}  "
              f"(n={int(r['report'][l]['support'])})")

    print()
    ba = r["balanced_accuracy"]
    if a.variant == "background":
        if ba > 0.90:
            print("  The repository is recoverable from the background alone,")
            print("  where no anatomy and therefore no pathology is present.")
            print("  The confound explanation -- that the probe is reading")
            print("  tumour phenotype -- does not account for this. Because")
            print("  'no tumour' is drawn exclusively from BR35H, that class")
            print("  carries an acquisition signature sufficient to identify it")
            print("  without reference to the brain.")
        elif ba > 0.70:
            print("  Moderate signal outside the head. Report the value and")
            print("  avoid strong claims in either direction.")
        else:
            print("  Little signal outside the head: the earlier whole-image")
            print("  probe was likely reading anatomy, not acquisition. The")
            print("  shortcut hypothesis is not supported by this test and")
            print("  should be withdrawn rather than hedged.")
    elif a.variant == "brain":
        print("  Compare against the background variant. Signal here is")
        print("  ambiguous: it may be anatomy, or residual acquisition")
        print("  differences within the head such as contrast and noise.")

    name = (f"{a.config}_{a.variant}{'_crop' if a.crop else ''}"
            f"{'_' + a.class_conditional if a.class_conditional else ''}"
            f"{'_' + '-'.join(a.pair) if a.pair else ''}")
    (OUT / f"{name}.json").write_text(json.dumps(
        {k: v for k, v in r.items() if k != "pred"}, indent=2))
    pd.DataFrame({"path": df.path, "source": df.source, "label": df.label,
                  "pred_source": r["pred"]}).to_csv(
        OUT / f"{name}_predictions.csv", index=False)
    print(f"\n  wrote {OUT / f'{name}.json'}")


if __name__ == "__main__":
    main()