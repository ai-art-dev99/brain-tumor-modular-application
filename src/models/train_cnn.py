#!/usr/bin/env python3
"""
train_cnn.py -- fine-tune EfficientNetB0 once per outer fold, and evaluate the
hybrid heads on the resulting fine-tuned features.

WHY PER-FOLD FINE-TUNING
------------------------
Fine-tuning once on all the data and then extracting features for every image
would put information from the test fold into the feature extractor. The
comparison "standalone CNN vs CNN+SVM" is only meaningful if both sides are
produced by a model that has never seen the images it is scored on. So the
backbone is retrained from ImageNet weights inside each outer fold, on that
fold's training portion alone.

WHAT THIS FIXES FROM THE ORIGINAL PIPELINE
------------------------------------------
- The best checkpoint was selected on the test set; here early stopping uses
  an inner validation fold drawn from the training portion.
- Augmentation (ColorJitter, flips, rotations) was applied to the evaluation
  loader, making every reported number a random draw; here evaluation uses a
  deterministic transform.
- The CNN and the hybrids were scored on different data; here both are scored
  on the same pooled out-of-fold predictions, so a paired test is valid.
- Feature extraction now yields the intended 1,280-d pooled vector rather than
  a 128,000-d flattened feature map.

A DOCUMENTED RESIDUAL OPTIMISM
------------------------------
The classical heads are tuned by cross-validation inside the outer training
portion, part of which the backbone saw during fine-tuning. Fully nesting the
backbone as well would require 5x3 fine-tuning runs. The test fold remains
untouched in either case, so the reported estimate is unbiased with respect to
it; the hyperparameter choice is mildly optimistic. State this in the
manuscript rather than leaving it implicit.

Usage
-----
    python train_cnn.py --config main --epochs 20
    python train_cnn.py --config main --heads svm logreg --epochs 20
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from train_eval import (full_metrics, group_bootstrap, model_grid,
                        paired_group_bootstrap)

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
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i], i


def build_transforms(cfg):
    """
    Training augmentation is intentionally mild. Rotation and scale jitter
    model realistic variation in head positioning; horizontal flipping is
    retained for consistency with prior work on this benchmark, though it is
    not strictly anatomy-preserving and its inclusion should be stated.
    Evaluation is fully deterministic.
    """
    size = cfg["input_size"][1]
    mean, std = cfg["mean"], cfg["std"]
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05),
                                scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    logits = torch.zeros(len(loader.dataset), n_classes)
    ys = torch.zeros(len(loader.dataset), dtype=torch.long)
    for xb, yb, idx in loader:
        with torch.autocast("cuda", enabled=device == "cuda"):
            out = model(xb.to(device, non_blocking=True))
        logits[idx] = out.float().cpu()
        ys[idx] = yb
    return logits, ys


@torch.no_grad()
def extract_features(model, loader, device, dim):
    """Pooled 1,280-d features from the fine-tuned backbone."""
    model.eval()
    feats = np.empty((len(loader.dataset), dim), dtype=np.float32)
    for xb, _, idx in loader:
        with torch.autocast("cuda", enabled=device == "cuda"):
            f = model.forward_features(xb.to(device, non_blocking=True))
            f = model.forward_head(f, pre_logits=True)
        feats[idx.numpy()] = f.float().cpu().numpy()
    return feats


def train_one_fold(train_df, val_df, labels, cfg, device, args, seed):
    n_classes = len(labels)
    lab2i = {l: i for i, l in enumerate(labels)}
    train_tf, eval_tf = build_transforms(cfg)

    tr_ds = MRIDataset(train_df.path, [lab2i[l] for l in train_df.label], train_tf)
    va_ds = MRIDataset(val_df.path, [lab2i[l] for l in val_df.label], eval_tf)
    g = torch.Generator().manual_seed(seed)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True,
                       drop_last=True, generator=g)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = timm.create_model("efficientnet_b0", pretrained=True,
                              num_classes=n_classes).to(device)

    # Inverse-frequency weighting rather than oversampling: duplicating images
    # inside a group would further amplify the few patients contributing many
    # slices.
    counts = train_df.label.value_counts().reindex(labels).to_numpy()
    w = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=0.05)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_score, best_state, bad, best_epoch = -1.0, None, 0, -1
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

        logits, ys = evaluate(model, va_dl, device, n_classes)
        pred = logits.argmax(1).numpy()
        # Balanced accuracy for selection: plain accuracy would let the model
        # coast on the easiest class.
        score = balanced_accuracy_score(ys.numpy(), pred)
        print(f"      epoch {ep + 1:2d}  loss {tot / len(tr_ds):.4f}  "
              f"val bal-acc {score:.4f}")

        if score > best_score:
            best_score, best_epoch, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"      early stop at epoch {ep + 1}")
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch + 1, "best_val_balanced_acc": best_score}


def run(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(SPLITS / f"splits_{args.config}_outer.csv")
    inner = pd.read_csv(SPLITS / f"splits_{args.config}_inner.csv")
    labels = sorted(df.label.unique())
    n_folds = df.outer_fold.nunique()

    run_id = args.tag or f"{args.config}_finetuned"
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    cfg = timm.data.resolve_model_data_config(probe)
    feat_dim = probe.num_features
    del probe
    _, eval_tf = build_transforms(cfg)

    print(f"==> {run_id}: {len(df)} images, {df.group_id.nunique()} groups, "
          f"{len(labels)} classes, {n_folds} folds")
    print(f"    device {device}, input {cfg['input_size']}, feature dim {feat_dim}")

    y = df.label.to_numpy()
    groups = df.group_id.to_numpy()
    folds = df.outer_fold.to_numpy()

    cnn_pred = np.empty(len(df), dtype=object)
    cnn_prob = np.zeros((len(df), len(labels)))
    head_pred = {h: np.empty(len(df), dtype=object) for h in args.heads}
    head_prob = {h: np.zeros((len(df), len(labels))) for h in args.heads}
    fold_info, t_start = [], time.time()

    for k in range(n_folds):
        print(f"\n--- outer fold {k}")
        tr_mask, te_mask = folds != k, folds == k
        tr_all = df[tr_mask].reset_index(drop=True)

        # Inner fold 0 is the early-stopping validation set for the backbone.
        sub = inner[inner.outer_fold == k]
        fold_of = dict(zip(sub.path, sub.inner_fold))
        tr_all["inner"] = tr_all.path.map(fold_of)
        fit_df = tr_all[tr_all.inner != 0]
        val_df = tr_all[tr_all.inner == 0]
        print(f"    fine-tune on {len(fit_df)}, validate on {len(val_df)}")

        model, info = train_one_fold(fit_df, val_df, labels, cfg, device,
                                     args, args.seed + k)

        # -- standalone CNN on the untouched test fold ------------------------
        te_df = df[te_mask].reset_index(drop=True)
        te_ds = MRIDataset(te_df.path, [labels.index(l) for l in te_df.label],
                           eval_tf)
        te_dl = DataLoader(te_ds, batch_size=args.batch_size * 2, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
        logits, _ = evaluate(model, te_dl, device, len(labels))
        prob = torch.softmax(logits, 1).numpy()
        cnn_prob[te_mask] = prob
        cnn_pred[te_mask] = np.array(labels)[prob.argmax(1)]
        acc = accuracy_score(y[te_mask], cnn_pred[te_mask])
        print(f"    CNN test accuracy {acc:.4f}")
        info["cnn_test_accuracy"] = float(acc)

        # -- hybrid heads on this fold's fine-tuned features -------------------
        if args.heads:
            all_tr = tr_all.reset_index(drop=True)
            tr_ds = MRIDataset(all_tr.path,
                               [labels.index(l) for l in all_tr.label], eval_tf)
            tr_dl = DataLoader(tr_ds, batch_size=args.batch_size * 2,
                               shuffle=False, num_workers=args.workers,
                               pin_memory=True)
            Xtr = extract_features(model, tr_dl, device, feat_dim)
            Xte = extract_features(model, te_dl, device, feat_dim)

            assign = all_tr.inner.to_numpy()
            cv = [(np.where(assign != m)[0], np.where(assign == m)[0])
                  for m in sorted(set(assign))]

            for h in args.heads:
                est, grid = model_grid(h, args.seed)
                pipe = Pipeline([("scale", StandardScaler()), ("clf", est)])
                gs = GridSearchCV(pipe, grid, cv=cv,
                                  scoring="balanced_accuracy", n_jobs=-1)
                gs.fit(Xtr, all_tr.label.to_numpy())
                head_pred[h][te_mask] = gs.predict(Xte)
                p = gs.predict_proba(Xte)
                col = [list(gs.best_estimator_.classes_).index(l) for l in labels]
                head_prob[h][te_mask] = p[:, col]
                a = accuracy_score(y[te_mask], head_pred[h][te_mask])
                print(f"    CNN+{h}: {a:.4f}   {gs.best_params_}")
                info[f"head_{h}_accuracy"] = float(a)
                info[f"head_{h}_params"] = {kk.replace("clf__", ""): vv
                                            for kk, vv in gs.best_params_.items()}

        if args.save_checkpoints:
            torch.save(model.state_dict(), out_dir / f"backbone_fold{k}.pt")
        fold_info.append(info)
        del model
        torch.cuda.empty_cache()

    # -- pooled metrics --------------------------------------------------------
    models_out = {}
    preds = {"path": df.path, "y_true": y, "group_id": groups,
             "outer_fold": folds}

    entries = [("cnn", cnn_pred.astype(str), cnn_prob)]
    entries += [(f"cnn_{h}", head_pred[h].astype(str), head_prob[h])
                for h in args.heads]

    print("\n" + "=" * 70)
    for name, p, pr in entries:
        m = full_metrics(y, p, pr, labels)
        for mn, fn in [("accuracy", accuracy_score),
                       ("balanced_accuracy", balanced_accuracy_score),
                       ("macro_f1", lambda a, b: f1_score(a, b, average="macro",
                                                          zero_division=0))]:
            m[f"{mn}_ci95"] = list(group_bootstrap(y, p, groups, fn,
                                                   args.bootstrap, args.seed))
        models_out[name] = m
        preds[f"pred_{name}"] = p
        for i, lab in enumerate(labels):
            preds[f"prob_{name}_{lab}"] = pr[:, i]
        print(f"  {name:<12} acc {m['accuracy']:.4f} "
              f"[{m['accuracy_ci95'][0]:.4f}, {m['accuracy_ci95'][1]:.4f}]  "
              f"bal {m['balanced_accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}")

    # -- the comparison the original manuscript could not make -----------------
    print("\n  paired comparisons (same images, same folds):")
    comparisons = []
    names = [e[0] for e in entries]
    for a, b in combinations(names, 2):
        pa, pb = preds[f"pred_{a}"], preds[f"pred_{b}"]
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
        "seed": args.seed, "n_images": len(df),
        "n_groups": int(df.group_id.nunique()), "labels": labels,
        "folds": fold_info, "models": models_out, "comparisons": comparisons,
        "total_seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)
    print(f"\n  wrote {out_dir}/  ({payload['total_seconds']:.0f}s total)")

    for name, m in models_out.items():
        print(f"\n  per-class -- {name}")
        print(pd.DataFrame(m["per_class"]).T[
            ["support", "precision", "recall_sensitivity", "specificity",
             "f1"]].round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="main")
    ap.add_argument("--heads", nargs="*", default=["svm", "logreg"])
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