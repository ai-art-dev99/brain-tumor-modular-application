#!/usr/bin/env python3
"""
train_cnn.py (v2) -- per-fold fine-tuning, evaluated against every head the
original manuscript compared.

CHANGES FROM v1
---------------
1. All four heads from the original study are now run on fine-tuned features:
   SVM, KNN, RF and MLP, with logistic regression as an additional baseline.
   v1 ran only SVM and logistic regression on fine-tuned features while KNN,
   RF and MLP were evaluated on frozen ImageNet features, so the revised paper
   could not be compared like-for-like with the original.

2. Head fitting is delegated to train_eval.fit_tuned, which disables
   scikit-learn's non-group-aware internal validation for the MLP and
   recalibrates the SVM over the group-disjoint inner folds. The deferral
   analysis depends on those probabilities.

3. Confidence intervals cover all metrics, and per-fold results are exported.

WHY THE BACKBONE IS RETRAINED INSIDE EVERY OUTER FOLD
-----------------------------------------------------
Fine-tuning once on all data and then extracting features for every image
would place test-fold information in the feature extractor, so "standalone CNN
vs CNN+SVM" would no longer be a comparison between two models that had never
seen the images they are scored on.

RESIDUAL OPTIMISM, STATED
-------------------------
The heads are tuned by cross-validation inside the outer training portion, and
the backbone saw part of that portion during fine-tuning. The outer test fold
is untouched either way, so the reported estimate is unbiased with respect to
it; the hyperparameter choice is mildly optimistic. Full nesting of the
backbone would require 5x3 fine-tuning runs and is a separate experiment.

Usage
-----
    python train_cnn.py --config main --epochs 20 --tag main_finetuned \\
        --save-checkpoints
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train_eval import (bootstrap_all, confusion_frame, fit_tuned,
                        metric_bundle, paired_group_bootstrap)

SPLITS = Path("/workspace/data/manifest/splits")
RUNS = Path("/workspace/outputs/runs")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MRIDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), \
               self.labels[i], i


def build_transforms(cfg):
    """Mild, anatomically plausible training augmentation; deterministic
    evaluation. Horizontal flipping is kept for continuity with prior work on
    this benchmark, though brain asymmetry means it is not strictly
    label-preserving -- state this in the manuscript."""
    s = cfg["input_size"][1]
    return (transforms.Compose([
        transforms.Resize((s, s)),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05),
                                scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]),
        transforms.Compose([
            transforms.Resize((s, s)), transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"])]))


@torch.no_grad()
def predict_logits(model, loader, device, n_classes):
    model.eval()
    out = torch.zeros(len(loader.dataset), n_classes)
    for xb, _, idx in loader:
        with torch.autocast("cuda", enabled=device == "cuda"):
            o = model(xb.to(device, non_blocking=True))
        out[idx] = o.float().cpu()
    return out


@torch.no_grad()
def extract_features(model, loader, device, dim):
    model.eval()
    f = np.empty((len(loader.dataset), dim), dtype=np.float32)
    for xb, _, idx in loader:
        with torch.autocast("cuda", enabled=device == "cuda"):
            z = model.forward_features(xb.to(device, non_blocking=True))
            z = model.forward_head(z, pre_logits=True)
        f[idx.numpy()] = z.float().cpu().numpy()
    return f


def train_one_fold(fit_df, val_df, labels, cfg, device, args, seed):
    n_cls = len(labels)
    lab2i = {l: i for i, l in enumerate(labels)}
    train_tf, eval_tf = build_transforms(cfg)
    g = torch.Generator().manual_seed(seed)

    tr_dl = DataLoader(MRIDataset(fit_df.path, [lab2i[l] for l in fit_df.label],
                                  train_tf),
                       batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True,
                       drop_last=True, generator=g)
    va_dl = DataLoader(MRIDataset(val_df.path, [lab2i[l] for l in val_df.label],
                                  eval_tf),
                       batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = timm.create_model("efficientnet_b0", pretrained=True,
                              num_classes=n_cls).to(device)
    counts = fit_df.label.value_counts().reindex(labels).to_numpy()
    w = torch.tensor(counts.sum() / (n_cls * counts), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best, best_state, bad, best_ep = -1.0, None, 0, -1
    yv = np.array([lab2i[l] for l in val_df.label])
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for xb, yb, _ in tr_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * xb.size(0)
        sched.step()
        pred = predict_logits(model, va_dl, device, n_cls).argmax(1).numpy()
        # Balanced accuracy for selection: plain accuracy lets the model coast
        # on whichever class is easiest, which here is the single-source one.
        sc = balanced_accuracy_score(yv, pred)
        print(f"      epoch {ep + 1:2d}  loss {tot / len(tr_dl.dataset):.4f}  "
              f"val bal-acc {sc:.4f}")
        if sc > best:
            best, best_ep, bad = sc, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"      early stop at epoch {ep + 1}")
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep + 1, "best_val_balanced_acc": float(best)}


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(SPLITS / f"splits_{args.config}_outer.csv")
    inner = pd.read_csv(SPLITS / f"splits_{args.config}_inner.csv")
    df["patient_id"] = df.patient_id.fillna("")
    labels = sorted(df.label.unique())
    n_folds = df.outer_fold.nunique()
    unit = "patient" if (df.patient_id != "").all() else "group"

    run_id = args.tag or f"{args.config}_finetuned"
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(tmp)
    feat_dim = tmp.num_features
    del tmp
    _, eval_tf = build_transforms(cfg)

    print(f"==> {run_id}: {len(df)} images, {df.group_id.nunique()} {unit}s, "
          f"{len(labels)} classes, {n_folds} folds")
    print(f"    device {device}, input {cfg['input_size']}, dim {feat_dim}")
    print(f"    heads: {', '.join(args.heads) if args.heads else '(none)'}")

    y, groups, folds = (df.label.to_numpy(), df.group_id.to_numpy(),
                        df.outer_fold.to_numpy())
    names = ["cnn"] + [f"cnn_{h}" for h in args.heads]
    pred = {n: np.empty(len(df), dtype=object) for n in names}
    prob = {n: np.zeros((len(df), len(labels))) for n in names}
    fold_info, per_fold, t0 = [], [], time.time()

    for k in range(n_folds):
        print(f"\n--- outer fold {k}")
        tr, te = folds != k, folds == k
        tr_all = df[tr].reset_index(drop=True)
        sub = inner[inner.outer_fold == k]
        tr_all["inner"] = tr_all.path.map(dict(zip(sub.path, sub.inner_fold)))
        fit_df, val_df = tr_all[tr_all.inner != 0], tr_all[tr_all.inner == 0]
        print(f"    fine-tune on {len(fit_df)}, early-stop on {len(val_df)}")

        model, info = train_one_fold(fit_df, val_df, labels, cfg, device,
                                     args, args.seed + k)

        te_df = df[te].reset_index(drop=True)
        te_dl = DataLoader(MRIDataset(te_df.path,
                                      [labels.index(l) for l in te_df.label],
                                      eval_tf),
                           batch_size=args.batch_size * 2, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
        lg = predict_logits(model, te_dl, device, len(labels))
        p = torch.softmax(lg, 1).numpy()
        prob["cnn"][te] = p
        pred["cnn"][te] = np.array(labels)[p.argmax(1)]
        print(f"    cnn          {accuracy_score(y[te], pred['cnn'][te]):.4f}")

        if args.heads:
            tr_dl = DataLoader(MRIDataset(tr_all.path,
                                          [labels.index(l) for l in tr_all.label],
                                          eval_tf),
                               batch_size=args.batch_size * 2, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
            Xtr = extract_features(model, tr_dl, device, feat_dim)
            Xte = extract_features(model, te_dl, device, feat_dim)
            a = tr_all.inner.to_numpy()
            cv = [(np.where(a != m)[0], np.where(a == m)[0]) for m in sorted(set(a))]

            for h in args.heads:
                est, params = fit_tuned(h, Xtr, tr_all.label.to_numpy(),
                                        cv, args.seed)
                nm = f"cnn_{h}"
                pred[nm][te] = est.predict(Xte)
                pp = est.predict_proba(Xte)
                col = [list(est.classes_).index(l) for l in labels]
                prob[nm][te] = pp[:, col]
                print(f"    {nm:<12} {accuracy_score(y[te], pred[nm][te]):.4f}"
                      f"   {params}")
                info[f"{nm}_params"] = {kk.replace("clf__", ""): vv
                                        for kk, vv in params.items()}

        for nm in names:
            fm = metric_bundle(y[te], pred[nm][te].astype(str), prob[nm][te], labels)
            per_fold.append({"model": nm, "fold": k, "n": int(te.sum()),
                             **{kk: vv for kk, vv in fm.items() if "::" not in kk}})

        if args.save_checkpoints:
            torch.save(model.state_dict(), out_dir / f"backbone_fold{k}.pt")
        fold_info.append(info)
        del model
        torch.cuda.empty_cache()

    # -- pooled ---------------------------------------------------------------
    print("\n" + "=" * 70)
    models_out, table = {}, {"path": df.path, "y_true": y,
                             "group_id": groups, "outer_fold": folds}
    for nm in names:
        pr = pred[nm].astype(str)
        m = metric_bundle(y, pr, prob[nm], labels)
        ci = bootstrap_all(y, pr, prob[nm], groups, labels, args.bootstrap, args.seed)
        models_out[nm] = {"point": m, "ci95": ci, "labels": labels,
                          "bootstrap_unit": unit,
                          "support": {l: int((y == l).sum()) for l in labels},
                          "confusion_matrix": confusion_frame(y, pr, labels)
                                              .values.tolist()}
        table[f"pred_{nm}"] = pr
        for i, lab in enumerate(labels):
            table[f"prob_{nm}_{lab}"] = prob[nm][:, i]
        lo, hi = ci.get("accuracy", [np.nan, np.nan])
        print(f"  {nm:<12} acc {m['accuracy']:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"bal {m['balanced_accuracy']:.4f}  F1 {m['macro_f1']:.4f}  "
              f"AUC {m.get('roc_auc_macro', float('nan')):.4f}  "
              f"Brier {m.get('brier', float('nan')):.4f}  "
              f"ECE {m.get('ece', float('nan')):.4f}")

    print(f"\n  paired comparisons ({unit}-level bootstrap is primary):")
    comparisons = []
    for a, b in combinations(names, 2):
        pa, pb = table[f"pred_{a}"], table[f"pred_{b}"]
        ca, cb = pa == y, pb == y
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        res = mcnemar(np.array([[int((ca & cb).sum()), n10],
                                [n01, int((~ca & ~cb).sum())]]), exact=True)
        d, lo, hi, pboot = paired_group_bootstrap(
            y, pa, pb, groups, accuracy_score, args.bootstrap, args.seed)
        comparisons.append({"model_a": a, "model_b": b,
                            "a_only_correct": n10, "b_only_correct": n01,
                            "mcnemar_exact_p": float(res.pvalue),
                            "acc_diff": d, "acc_diff_ci95": [lo, hi],
                            "bootstrap_p": pboot})
        print(f"    {a} vs {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"McNemar p={res.pvalue:.4f}  bootstrap p={pboot:.4f}")

    payload = {
        "run_id": run_id, "config": args.config, "split_mode": "grouped",
        "backbone": "efficientnet_b0 (timm, ImageNet init, fully fine-tuned)",
        "input_size": list(cfg["input_size"]), "feature_dim": feat_dim,
        "optimiser": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "max_epochs": args.epochs,
        "patience": args.patience, "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropy, inverse-frequency weights, label_smoothing=0.05",
        "augmentation": "RandomAffine(15deg, 5% translate, 0.9-1.1 scale), "
                        "HFlip(0.5), ColorJitter(0.15/0.15); eval deterministic",
        "seed": args.seed, "n_images": len(df),
        "n_groups": int(df.group_id.nunique()), "bootstrap_unit": unit,
        "labels": labels, "folds": fold_info, "models": models_out,
        "comparisons": comparisons,
        "total_seconds": round(time.time() - t0, 1)}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(table).to_csv(out_dir / "predictions.csv", index=False)
    pf = pd.DataFrame(per_fold)
    pf.to_csv(out_dir / "per_fold.csv", index=False)

    print("\n  between-fold spread:")
    print(pf.groupby("model").accuracy.agg(["min", "mean", "max", "std"])
            .round(4).to_string())

    for nm, v in models_out.items():
        print(f"\n  per-class -- {nm}")
        t = pd.DataFrame({met: {l: v["point"].get(f"{met}::{l}", np.nan)
                                for l in labels}
                          for met in ["precision", "recall", "specificity",
                                      "f1", "auc"]})
        t["support"] = pd.Series(v["support"])
        print(t.round(3).to_string())

    print(f"\n  wrote {out_dir}/  ({payload['total_seconds']:.0f}s)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="main")
    ap.add_argument("--heads", nargs="*",
                    default=["svm", "knn", "rf", "mlp", "logreg"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-checkpoints", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()