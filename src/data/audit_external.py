#!/usr/bin/env python3
"""
audit_external.py (v3) -- screen a candidate external dataset before any model
sees it.

NEW IN v3: THE CANDIDATE'S OWN SPLIT
------------------------------------
Datasets distributed with train/val/test folders make a claim about those
folders: that a model trained on one can be honestly evaluated on another.
That claim is testable with the same near-duplicate clustering used
internally, and on this benchmark family it has already failed twice. So the
audit now:

  - records which published partition each image came from,
  - reports whether near-duplicate groups straddle those partitions, which is
    exactly the defect that inflates published accuracy on this benchmark,
  - reports overlap with our development data per partition, so a decision to
    use only the published test subset can be made on evidence.

A candidate that leaks internally is still usable as an external cohort for
us -- we never train on it -- but the leakage is worth reporting, because it
bears on every result others have published using that split.

THREE SCOPES (unchanged from v2)
--------------------------------
  trained    the 3,813 images that entered our experiments. A match means the
             model would be tested on something it has seen.
  sources    Figshare, SARTAJ and BR35H before deduplication, excluding the
             composite. A match means shared source family, which is what
             "independent source-based test set" rules out.
  composite  the Nickparvar aggregate, reported separately so it is never
             mistaken for a fourth acquisition source.

Overlap is counted over near-duplicate groups, not files, and confirmed with
SSIM on a common grid so the claim rests on two independent measures.

Usage
-----
    python audit_external.py --name bdneuro_v7 --root "/path/to/dataset"
    python audit_external.py --name bdneuro_v7 --root ... --restrict-split test
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import hashlib
import json
import re
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

MANIFEST = Path("/workspace/data/manifest")
EXTERNAL = Path("/workspace/data/external")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_ALIASES = {
    "glioma": "glioma", "brain_glioma": "glioma", "glioma_tumor": "glioma",
    "meningioma": "meningioma", "brain_menin": "meningioma",
    "meningioma_tumor": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary",
    "pituitary_macroadenoma": "pituitary",
    "notumor": "notumor", "no_tumor": "notumor", "normal": "notumor",
    "brain_normal": "notumor", "no": "notumor", "healthy": "notumor",
    "nontumor": "notumor", "non_tumor": "notumor",
}

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
SSIM_SIZE = 256


def sha256_of(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def hex_to_packed(s) -> np.ndarray:
    return np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in s],
                    dtype=np.uint8)


def dual_nearest(pa, da, pb, db, chunk: int = 256):
    bi = np.zeros(len(pa), dtype=np.int64)
    bd = np.full(len(pa), 127, dtype=np.int16)
    for s in range(0, len(pa), chunk):
        e = min(s + chunk, len(pa))
        dp = POPCOUNT[pa[s:e, None, :] ^ pb[None, :, :]].sum(axis=2)
        dd = POPCOUNT[da[s:e, None, :] ^ db[None, :, :]].sum(axis=2)
        d = np.maximum(dp, dd).astype(np.int16)
        bi[s:e], bd[s:e] = d.argmin(axis=1), d.min(axis=1)
    return bi, bd


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _norm(part: str) -> str:
    k = part.strip().lower().replace(" ", "_").replace("-", "_")
    # Some releases prefix class folders with the image size, e.g. "512Glioma".
    return re.sub(r"^\d+[_\-]?", "", k)


def normalise_class(rel: Path) -> str:
    for part in reversed(rel.parts):
        if (k := _norm(part)) in CLASS_ALIASES:
            return CLASS_ALIASES[k]
    return "unknown"


def detect_split(rel: Path) -> str:
    """The partition the publisher shipped the image in, if any."""
    for part in rel.parts:
        if (k := _norm(part)) in SPLIT_ALIASES:
            return SPLIT_ALIASES[k]
    return "unassigned"


def index_external(root: Path) -> pd.DataFrame:
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise SystemExit(f"no images under {root}")
    print(f"==> indexing {len(paths)} files under {root}")
    rows = []
    for p in tqdm(paths, unit="img"):
        rel = p.relative_to(root)
        try:
            with Image.open(p) as im:
                w, h = im.size
                g = im.convert("L")
                ph, dh = imagehash.phash(g, 8), imagehash.dhash(g, 8)
        except Exception as e:
            print(f"    unreadable, skipped: {p} ({e})")
            continue
        rows.append({"path": str(p), "rel": str(rel),
                     "label": normalise_class(rel),
                     "published_split": detect_split(rel),
                     "width": w, "height": h, "bytes": p.stat().st_size,
                     "sha256": sha256_of(p), "phash": str(ph), "dhash": str(dh)})
    return pd.DataFrame(rows)


def load_scope(scope: str) -> pd.DataFrame:
    files = pd.read_csv(MANIFEST / "files_index.csv")
    fig = pd.read_csv(MANIFEST / "figshare_index.csv")
    fig_rows = pd.DataFrame({"path": fig.render_path, "source": "figshare",
                             "class": fig["class"], "phash": fig.phash,
                             "dhash": fig.dhash, "sha256": ""})
    cols = ["path", "source", "class", "phash", "dhash", "sha256"]

    if scope == "trained":
        ds = pd.read_csv(MANIFEST / "dataset.csv")
        pool = pd.concat([files[cols], fig_rows], ignore_index=True)
        return pool[pool.path.isin(set(ds.path))].reset_index(drop=True)
    if scope == "sources":
        f = files[files.source.isin(["sartaj", "br35h"])]
        return pd.concat([f[cols], fig_rows], ignore_index=True)
    if scope == "composite":
        return files[files.source == "aggregated"][cols].reset_index(drop=True)
    raise SystemExit(f"unknown scope {scope}")


def load_grey(p: str):
    try:
        return np.asarray(Image.open(p).convert("L")
                          .resize((SSIM_SIZE, SSIM_SIZE)), dtype=np.float32)
    except Exception:
        return None


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--group-threshold", type=int, default=2)
    ap.add_argument("--restrict-split", default=None,
                    choices=["train", "val", "test"],
                    help="audit only one published partition")
    ap.add_argument("--ssim-sample", type=int, default=400)
    ap.add_argument("--published-total", type=int, default=None)
    ap.add_argument("--montage", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    EXTERNAL.mkdir(parents=True, exist_ok=True)
    ext = index_external(Path(a.root))

    if a.restrict_split:
        before = len(ext)
        ext = ext[ext.published_split == a.restrict_split].reset_index(drop=True)
        print(f"\n  restricted to published '{a.restrict_split}' partition: "
              f"{len(ext)} of {before} images")
    n_raw = len(ext)

    # -- composition as distributed -------------------------------------------
    print("\n" + "=" * 74)
    print("Composition as distributed")
    print("=" * 74)
    print(f"  files on disk: {n_raw}")
    print(pd.crosstab(ext.label, ext.published_split, margins=True).to_string())
    if (ext.label == "unknown").any():
        folders = sorted({Path(r).parent.name for r in
                          ext.rel[ext.label == "unknown"]})
        print(f"\n  WARNING: {(ext.label == 'unknown').sum()} images unmapped. "
              f"Folders: {folders[:10]}")
    if a.published_total and n_raw != a.published_total:
        print(f"\n  DISCREPANCY: {n_raw} files distributed against "
              f"{a.published_total} published.\n  Both counts are of files as "
              f"shipped; report both rather than substituting one.")

    dup_exact = ext[ext.duplicated("sha256", keep=False)]
    print(f"\n  byte-identical duplicate files: {len(dup_exact)} in "
          f"{dup_exact.sha256.nunique()} groups")
    dims = ext.width.astype(str) + "x" + ext.height.astype(str)
    print(f"  distinct dimensions: {dims.nunique()} "
          f"(most common {dims.value_counts().index[0]})")

    # -- near-duplicate grouping ----------------------------------------------
    ph, dh = hex_to_packed(ext.phash), hex_to_packed(ext.dhash)
    dsu = DSU(len(ext))
    for s in range(0, len(ext), 256):
        e = min(s + 256, len(ext))
        dp = POPCOUNT[ph[s:e, None, :] ^ ph[None, :, :]].sum(axis=2)
        dd = POPCOUNT[dh[s:e, None, :] ^ dh[None, :, :]].sum(axis=2)
        r, c = np.where((dp <= a.group_threshold) & (dd <= a.group_threshold))
        for i, j in zip(r, c):
            gi, gj = s + int(i), int(j)
            if gi < gj:
                dsu.union(gi, gj)
    ext["group_id"] = [f"{a.name}_g{dsu.find(i):05d}" for i in range(len(ext))]
    n_groups = ext.group_id.nunique()
    print(f"  near-duplicate groups (pseudo-patients): {n_groups} "
          f"({n_raw / n_groups:.2f} images per group)")

    # -- does the publisher's own split leak? ---------------------------------
    split_report = None
    if ext.published_split.nunique() > 1 and not a.restrict_split:
        print("\n" + "=" * 74)
        print("Integrity of the publisher's own train/val/test split")
        print("=" * 74)
        spread = ext.groupby("group_id").published_split.nunique()
        n_span = int((spread > 1).sum())
        span_ids = set(spread[spread > 1].index)
        n_imgs = int(ext.group_id.isin(span_ids).sum())
        print(f"  near-duplicate groups spanning more than one partition: "
              f"{n_span} / {n_groups} ({100 * n_span / n_groups:.1f}%)")
        print(f"  images involved: {n_imgs} ({100 * n_imgs / n_raw:.1f}%)")

        te = ext[ext.published_split == "test"]
        if len(te):
            tr_groups = set(ext[ext.published_split != "test"].group_id)
            n_bad = int(te.group_id.isin(tr_groups).sum())
            print(f"  test images with a near-duplicate in train or val: "
                  f"{n_bad} / {len(te)} ({100 * n_bad / len(te):.1f}%)")
        if n_span:
            print("  -> the published split is not free of near-duplicates. "
                  "This does not\n     affect our use of the data, since we "
                  "never train on it, but it does\n     bear on results others "
                  "have reported using this split.")
        else:
            print("  -> no near-duplicate group straddles the published "
                  "partitions.")
        split_report = {"groups_spanning_splits": n_span,
                        "images_involved": n_imgs,
                        "test_images_with_sibling": int(n_bad) if len(te) else None}

    # -- overlap with our data, one scope at a time ---------------------------
    results, per_scope = {}, {}
    for scope in ["trained", "sources", "composite"]:
        dev = load_scope(scope)
        if not len(dev):
            continue
        print("\n" + "=" * 74)
        print(f"Scope '{scope}': {len(dev)} development images")
        print("=" * 74)
        bi, bd = dual_nearest(ph, dh, hex_to_packed(dev.phash),
                              hex_to_packed(dev.dhash))
        hit = bd <= a.threshold
        ext[f"{scope}_distance"] = bd
        ext[f"{scope}_match_path"] = dev.path.to_numpy()[bi]
        ext[f"{scope}_match_source"] = dev.source.to_numpy()[bi]
        ext[f"{scope}_match_class"] = dev["class"].to_numpy()[bi]
        ext[f"{scope}_overlap"] = hit

        n_exact = len(set(ext.sha256) & set(dev.sha256[dev.sha256 != ""]))
        gflag = ext.groupby("group_id")[f"{scope}_overlap"].any()
        g_hit = int(gflag.sum())
        print(f"  byte-identical matches: {n_exact}")
        print(f"  images at dual-hash <= {a.threshold}: {int(hit.sum())} / "
              f"{n_raw} ({100 * hit.mean():.1f}%)")
        print(f"  GROUPS contaminated: {g_hit} / {n_groups} "
              f"({100 * g_hit / n_groups:.1f}%)")
        print(f"  minimum distance observed: {int(bd.min())}")

        print("\n  distance distribution (first 16 bins):")
        hist = pd.Series(bd).value_counts().sort_index()
        mx = hist.max()
        for dval, n in hist.head(16).items():
            print(f"    {dval:3d} | {n:5d} {'#' * min(52, int(52 * n / mx))}")

        print("\n  overlap by class and published partition:")
        print(pd.crosstab([ext.label, ext.published_split],
                          ext[f"{scope}_overlap"]).to_string())

        if hit.any():
            print("\n  matched images by external class and development source:")
            print(pd.crosstab(ext.label[hit],
                              ext[f"{scope}_match_source"][hit]).to_string())
            agree = (ext.label[hit] == ext[f"{scope}_match_class"][hit]).mean()
            print(f"\n  {100 * agree:.1f}% of matched pairs share a diagnostic "
                  f"label (chance ~{100 / max(ext.label.nunique(), 1):.0f}%)")

        results[scope] = {
            "development_images": len(dev), "byte_identical": n_exact,
            "images_matched": int(hit.sum()), "image_rate": float(hit.mean()),
            "groups_contaminated": g_hit,
            "group_rate": float(g_hit / n_groups),
            "min_distance": int(bd.min())}
        per_scope[scope] = gflag

    # -- SSIM confirmation -----------------------------------------------------
    conf = "sources" if "sources" in results else next(iter(results), None)
    if a.ssim_sample and conf:
        idx = np.where(ext[f"{conf}_overlap"].to_numpy())[0]
        if len(idx):
            rng = np.random.default_rng(a.seed)
            take = rng.choice(idx, min(a.ssim_sample, len(idx)), replace=False)
            print("\n" + "=" * 74)
            print(f"SSIM confirmation on {len(take)} matched pairs "
                  f"(scope '{conf}')")
            print("=" * 74)
            v, d = [], []
            for i in tqdm(take, unit="pair"):
                A = load_grey(ext.path.iloc[i])
                B = load_grey(ext[f"{conf}_match_path"].iloc[i])
                if A is None or B is None:
                    continue
                v.append(ssim(A, B, data_range=255.0))
                d.append(int(ext[f"{conf}_distance"].iloc[i]))
            v, d = np.array(v), np.array(d)
            print(f"  SSIM median {np.median(v):.3f}  mean {v.mean():.3f}  "
                  f"min {v.min():.3f}")
            for lo, hi in [(0, .5), (.5, .8), (.8, .9), (.9, .95), (.95, 1.01)]:
                m = (v >= lo) & (v < hi)
                if m.sum():
                    print(f"    SSIM [{lo:.2f}, {hi:.2f}): {int(m.sum()):4d} "
                          f"pairs, mean hash distance {d[m].mean():.1f}")
            print(f"\n  {100 * (v >= 0.95).mean():.1f}% of pairs reach "
                  f"SSIM >= 0.95. Hash distance and SSIM are\n  independent "
                  f"measures; their agreement is what makes the claim hold.")
            results["ssim"] = {"scope": conf, "n": int(len(v)),
                               "median": float(np.median(v)),
                               "min": float(v.min()),
                               "frac_above_0.95": float((v >= 0.95).mean())}
        else:
            print("\n  no matched pairs to confirm with SSIM.")

    # -- what would remain -----------------------------------------------------
    if "sources" in per_scope:
        keep = set(per_scope["sources"][~per_scope["sources"]].index)
        kept = ext[ext.group_id.isin(keep)]
        print("\n" + "=" * 74)
        print("Cohort remaining after excluding contaminated groups "
              "(scope 'sources')")
        print("=" * 74)
        s = pd.DataFrame({"distributed": ext.label.value_counts(),
                          "retained": kept.label.value_counts()}
                         ).fillna(0).astype(int)
        s["removed"] = s.distributed - s.retained
        s["retained_%"] = (100 * s.retained / s.distributed).round(1)
        print(s.to_string())
        if len(kept) and "test" in set(ext.published_split):
            print("\n  retained, restricted to the published test partition:")
            print(kept[kept.published_split == "test"].label
                  .value_counts().to_string())
        print()
        empty = s[s.retained == 0]
        if len(empty):
            print(f"  Classes reduced to zero: {', '.join(empty.index)}. A "
                  f"four-class external\n  evaluation cannot be constructed at "
                  f"any filtering level.")
        elif s.retained.sum() < 0.5 * n_raw:
            print("  More than half the cohort is excluded. What remains is of")
            print("  unestablished rather than established provenance:")
            print("  perceptual hashing finds detectable reuse, not all reuse.")
        elif results.get("sources", {}).get("group_rate", 1) < 0.01:
            print("  Contamination is negligible. This cohort is a defensible")
            print("  external test set. Lock the protocol and the model before")
            print("  running inference, and report the screening alongside it.")
        else:
            print("  A decontaminated cohort is available. State the exclusion")
            print("  counts and describe it as a secondary cross-dataset")
            print("  analysis, not a fully independent external cohort.")
        results["retained_cohort"] = s.to_dict()

    if a.montage and conf:
        write_montages(ext, a.name, conf, a.montage)

    ext.to_csv(EXTERNAL / f"{a.name}_manifest.csv", index=False)
    (EXTERNAL / f"{a.name}_overlap_report.json").write_text(json.dumps({
        "name": a.name, "root": str(a.root),
        "restricted_split": a.restrict_split,
        "files": n_raw,
        "class_by_split": pd.crosstab(ext.label,
                                      ext.published_split).to_dict(),
        "exact_duplicate_files": int(len(dup_exact)),
        "near_duplicate_groups": int(n_groups),
        "published_split_integrity": split_report,
        "overlap_threshold": a.threshold,
        "scopes": results}, indent=2, default=str))
    print(f"\n  wrote {EXTERNAL / f'{a.name}_manifest.csv'}")
    print(f"  wrote {EXTERNAL / f'{a.name}_overlap_report.json'}")


def write_montages(ext, name, scope, n, T=240):
    d = EXTERNAL / f"{name}_pairs_{scope}"
    hit = ext[ext[f"{scope}_overlap"]]
    if not len(hit):
        print("\n  no matched pairs; no montages written.")
        return
    d.mkdir(parents=True, exist_ok=True)
    per = max(1, n // 3)
    picks = []
    for lo, hi in [(0, 0), (1, 2), (3, 99)]:
        b = hit[(hit[f"{scope}_distance"] >= lo) &
                (hit[f"{scope}_distance"] <= hi)]
        if len(b):
            picks.append(b.sample(min(per, len(b)), random_state=0))
    for rank, (_, r) in enumerate(pd.concat(picks).iterrows(), 1):
        sheet = Image.new("RGB", (T * 2, T), "black")
        for k, p in enumerate([r.path, r[f"{scope}_match_path"]]):
            try:
                sheet.paste(Image.open(p).convert("RGB").resize((T, T)), (k * T, 0))
            except Exception:
                pass
        sheet.save(d / f"pair{rank:02d}_d{int(r[f'{scope}_distance'])}"
                      f"_{r.label}_vs_{r[f'{scope}_match_source']}.png")
    print(f"\n  wrote montages to {d} (left external, right development)")


if __name__ == "__main__":
    main()