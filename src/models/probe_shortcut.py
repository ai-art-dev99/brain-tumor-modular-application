#!/usr/bin/env python3
"""
probe_shortcut.py (v2) -- is repository identity recoverable from
non-diagnostic image content?

CHANGES FROM v1
---------------
1. The interpretation text was written for the whole-dataset run and printed
   verbatim in every mode. After a meningioma-only run it claimed that
   "'no tumour' is drawn exclusively from BR35H", which is true of the dataset
   but irrelevant to a run containing neither BR35H nor the no-tumour class.
   Conclusions are now derived from what the run actually contains.

2. "no anatomy and therefore no pathology is present" overstated the masking.
   The region outside an Otsu head mask still carries the skull contour, the
   field of view, cropping and padding decisions, resampling and JPEG
   artefacts. What the probe establishes is source-specific NON-PATHOLOGICAL
   signal, which is the claim that matters and is defensible; scanner
   acquisition alone is not separable from storage pipeline here.

3. Adjectival verdicts ("moderate signal") are gone. The script reports
   balanced accuracy against the chance level and the majority baseline, plus
   per-source recall, and states what does and does not follow. The reader
   draws the line.

VARIANTS
--------
  full        whole image; source and diagnosis are confounded.
  background  everything outside the estimated head mask. No brain tissue, so
              tumour phenotype cannot explain a positive result.
  brain       inside the mask only; --crop additionally crops to the head
              bounding box and rescales, removing framing and field-of-view
              cues along with the background.

Usage
-----
    python probe_shortcut.py --variant background --pair figshare br35h
    python probe_shortcut.py --variant background --class-conditional meningioma
    python probe_shortcut.py --variant brain --crop
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
    try:
        t = threshold_otsu(grey)
    except Exception:
        t = grey.mean()
    m = ndimage.binary_fill_holes(grey > t)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    if m.sum() < 0.02 * m.size:
        m = np.ones_like(m, dtype=bool)
    return m.astype(bool)


class MaskedImages(Dataset):
    def __init__(self, paths, variant, crop, size, mean, std):
        self.paths, self.variant, self.crop, self.size = list(paths), variant, crop, size
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = np.asarray(Image.open(self.paths[i]).convert("L")
                       .resize((self.size, self.size)), dtype=np.uint8)
        if self.variant != "full":
            m = head_mask(a.astype(np.float32))
            if self.variant == "brain":
                a = np.where(m, a, 0).astype(np.uint8)
                if self.crop and m.any():
                    ys, xs = np.where(m)
                    a = np.asarray(
                        Image.fromarray(a[ys.min():ys.max() + 1,
                                          xs.min():xs.max() + 1])
                        .resize((self.size, self.size)), dtype=np.uint8)
            else:
                a = np.where(m, 0, a).astype(np.uint8)
        return self.norm(Image.fromarray(np.repeat(a[:, :, None], 3, 2))), i


@torch.no_grad()
def features_for(paths, variant, crop, batch_size, workers):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0,
                              global_pool="avg").eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    ds = MaskedImages(paths, variant, crop, cfg["input_size"][1],
                      cfg["mean"], cfg["std"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)
    out = np.empty((len(ds), model.num_features), dtype=np.float32)
    tag = f"{variant}{'+crop' if crop else ''}"
    for xb, idx in tqdm(dl, unit="batch", desc=tag):
        out[idx.numpy()] = model(xb.to(device)).float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out


def probe(X, y_src, folds, seed):
    """Out-of-fold source classification on the diagnostic splits, so no
    patient contributes to both sides."""
    preds = np.empty(len(y_src), dtype=object)
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        if len(set(y_src[tr])) < 2:
            preds[te] = y_src[tr][0]
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000,
                                               class_weight="balanced",
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


def conclusions(r: dict, variant: str, crop: bool, df: pd.DataFrame,
                class_fixed: str | None, restricted: bool = False) -> list[str]:
    """
    What follows from THIS run. Everything below is conditioned on the data
    actually present, not on the dataset as a whole.
    """
    ba = r["balanced_accuracy"]
    chance = 1.0 / len(r["labels"])
    lines = [f"balanced accuracy {ba:.4f} against a chance level of {chance:.4f}"]

    single_source = df.groupby("label").source.nunique().eq(1)
    single = list(single_source[single_source].index)

    if variant == "background":
        lines.append(
            "The probe saw only the region outside an Otsu head-mask estimate. "
            "That region\n  excludes brain tissue, so tumour phenotype cannot "
            "account for the result. It does\n  retain the skull contour, field "
            "of view, cropping, padding, resampling and\n  compression, so what "
            "is demonstrated is source-specific NON-PATHOLOGICAL signal --\n  "
            "not scanner acquisition in isolation, which these repositories do "
            "not let us separate.")
        if class_fixed:
            lines.append(
                f"The diagnostic class is held constant at '{class_fixed}', so "
                f"the confound in which\n  source and pathology co-vary is "
                f"removed by design. A high value here cannot be\n  explained by "
                f"the probe reading tumour type.")
        elif single:
            # Restricting to a pair of repositories can make a class look
            # single-source when it is not in the full dataset -- meningioma
            # and pituitary both draw on Figshare and SARTAJ. Say which is
            # meant.
            scope = (" within this restricted subset" if restricted
                     else " in the full dataset")
            lines.append(
                f"Classes drawn from a single repository{scope}: "
                f"{', '.join(single)}. For those\n  classes the label is "
                f"recoverable from source identity alone, so their per-class\n  "
                f"figures cannot be read as diagnostic performance without "
                f"qualification.")
        else:
            lines.append(
                "No class in this run comes from a single repository, so source "
                "identity does not\n  by itself determine any label here.")
    elif variant == "brain":
        lines.append(
            "Background was removed" + (" and the head cropped and rescaled"
                                        if crop else "") +
            ", so framing cues are reduced. Signal remaining\n  here is "
            "ambiguous between anatomy and within-head acquisition differences "
            "such as\n  sequence, contrast and noise. Read it against the "
            "background variant: if both are\n  high, masking the background "
            "does not remove the confound.")
    else:
        lines.append(
            "Whole-image probe: source and diagnosis are confounded in this "
            "benchmark, so this\n  value alone does not distinguish acquisition "
            "signature from tumour phenotype. Use\n  the background and "
            "class-conditional variants for that.")

    worst = min(r["report"][l]["recall"] for l in r["labels"])
    if worst < 0.80:
        low = [l for l in r["labels"] if r["report"][l]["recall"] < 0.80]
        lines.append(
            f"Recall is below 0.80 for {', '.join(low)}, which pulls the "
            f"balanced figure down.\n  Where one repository is a "
            f"re-encoded derivative of another, confusion between the\n  two is "
            f"expected and is itself evidence of that relationship; report the "
            f"pairwise\n  comparison alongside the multi-way one.")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="main")
    ap.add_argument("--variant", choices=["full", "brain", "background"],
                    default="background")
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--pair", nargs=2, default=None, metavar=("SRC_A", "SRC_B"))
    ap.add_argument("--class-conditional", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SPLITS / f"splits_{a.config}_outer.csv")
    if a.class_conditional:
        df = df[df.label == a.class_conditional]
        print(f"==> diagnosis held constant: '{a.class_conditional}'")
    if a.pair:
        df = df[df.source.isin(a.pair)]
        print(f"==> restricted to {a.pair[0]} vs {a.pair[1]}")
    df = df.reset_index(drop=True)

    counts = df.source.value_counts()
    print(f"\n  {len(df)} images")
    print(counts.to_string())
    if len(counts) < 2:
        raise SystemExit("fewer than two sources remain; nothing to probe.")
    print(f"\n  classes present: {', '.join(sorted(df.label.unique()))}")
    print(f"  majority-source baseline accuracy: {counts.max() / len(df):.4f}")
    print(f"  chance balanced accuracy:          {1 / len(counts):.4f}")
    if counts.min() < 30:
        print(f"  WARNING: smallest source has {counts.min()} images; the "
              f"estimate is unstable\n           regardless of its value.")

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

    print("\n  What follows from this run:")
    restricted = bool(a.pair or a.class_conditional)
    for line in conclusions(r, a.variant, a.crop, df, a.class_conditional,
                            restricted):
        print(f"  - {line}")

    name = (f"{a.config}_{a.variant}{'_crop' if a.crop else ''}"
            f"{'_' + a.class_conditional if a.class_conditional else ''}"
            f"{'_' + '-'.join(a.pair) if a.pair else ''}")
    payload = {k: v for k, v in r.items() if k != "pred"}
    payload.update({"variant": a.variant, "crop": a.crop,
                    "class_conditional": a.class_conditional,
                    "pair": a.pair, "n_images": len(df),
                    "source_counts": counts.to_dict(),
                    "classes_present": sorted(df.label.unique()),
                    "majority_baseline": float(counts.max() / len(df)),
                    "chance_balanced": float(1 / len(counts))})
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame({"path": df.path, "source": df.source, "label": df.label,
                  "pred_source": r["pred"]}).to_csv(
        OUT / f"{name}_predictions.csv", index=False)
    print(f"\n  wrote {OUT / f'{name}.json'}")


if __name__ == "__main__":
    main()