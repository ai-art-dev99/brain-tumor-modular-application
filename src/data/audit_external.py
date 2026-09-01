#!/usr/bin/env python3
"""
audit_external.py (v2) -- screen a candidate external dataset before any model
sees it.

THREE QUESTIONS, THREE SCOPES
-----------------------------
v1 pooled every development image into one comparison, which conflated claims
that need to be kept apart. The `aggregated` rows are the Nickparvar composite
Kaggle benchmark, itself assembled from Figshare, SARTAJ and BR35H; those
images were never used for training here, so a match against them is evidence
about provenance, not contamination of this study.

  trained    the 3,813 images that actually entered the experiments.
             A match here means the model was tested on something it saw.

  sources    the original repositories (Figshare, SARTAJ, BR35H) before
             deduplication, excluding the composite. A match here means the
             candidate is drawn from the same source family, which is what
             "genuinely independent source-based test set" rules out, even if
             that particular image never entered training.

  composite  the Nickparvar aggregate. Reported separately so it is never
             mistaken for a fourth acquisition source.

GROUP-LEVEL ACCOUNTING
----------------------
Overlap is counted over near-duplicate clusters, not files. Three re-encodings
of one scan matching one development image are one contaminated unit, not
three. A cluster is excluded whole if any member matches.

SSIM CONFIRMATION
-----------------
A perceptual hash collision is evidence, not proof. Structural similarity is
computed for every matched pair on a common grid, so the claim rests on two
independent measures. Report the joint distribution rather than a single
threshold.

COUNT REPORTING
---------------
Raw file counts are reported before any collapsing, then exact duplicates,
then near-duplicate groups. Where a repository's published composition differs
from the files it distributes, both numbers are stated; neither is silently
substituted for the other.

Usage
-----
    python audit_external.py --name pmram --root "/path/to/Raw"
    python audit_external.py --name pmram --root ... --ssim-sample 400
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
    "notumor": "notumor", "no_tumor": "notumor", "normal": "notumor",
    "brain_normal": "notumor", "no": "notumor", "healthy": "notumor",
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
    """Nearest neighbour under max(pHash, dHash) Hamming distance."""
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


def normalise_class(path: Path) -> str:
    for part in reversed(path.parts):
        k = part.strip().lower().replace(" ", "_").replace("-", "_")
        # Some releases prefix class folders with the image size, e.g. PMRAM's
        # "512Glioma". Without stripping it every image falls through.
        k = re.sub(r"^\d+[_\-]?", "", k)
        if k in CLASS_ALIASES:
            return CLASS_ALIASES[k]
    return "unknown"


def index_external(root: Path) -> pd.DataFrame:
    paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise SystemExit(f"no images under {root}")
    print(f"==> indexing {len(paths)} files under {root}")
    rows = []
    for p in tqdm(paths, unit="img"):
        try:
            with Image.open(p) as im:
                w, h = im.size
                g = im.convert("L")
                ph, dh = imagehash.phash(g, 8), imagehash.dhash(g, 8)
        except Exception as e:
            print(f"    unreadable, skipped: {p} ({e})")
            continue
        rows.append({"path": str(p), "rel": str(p.relative_to(root)),
                     "label": normalise_class(p.relative_to(root)),
                     "width": w, "height": h, "bytes": p.stat().st_size,
                     "sha256": sha256_of(p), "phash": str(ph), "dhash": str(dh)})
    return pd.DataFrame(rows)


def load_scope(scope: str) -> pd.DataFrame:
    """Development images for one comparison scope."""
    files = pd.read_csv(MANIFEST / "files_index.csv")
    fig = pd.read_csv(MANIFEST / "figshare_index.csv")
    fig_rows = pd.DataFrame({"path": fig.render_path, "source": "figshare",
                             "class": fig["class"], "phash": fig.phash,
                             "dhash": fig.dhash, "sha256": ""})

    if scope == "trained":
        # Exactly what entered the experiments. dataset.csv carries the
        # representative paths; hashes are joined back from the indices.
        ds = pd.read_csv(MANIFEST / "dataset.csv")
        pool = pd.concat([files[["path", "source", "class", "phash", "dhash",
                                 "sha256"]], fig_rows], ignore_index=True)
        return pool[pool.path.isin(set(ds.path))].reset_index(drop=True)

    if scope == "sources":
        f = files[files.source.isin(["sartaj", "br35h"])]
        return pd.concat([f[["path", "source", "class", "phash", "dhash",
                             "sha256"]], fig_rows], ignore_index=True)

    if scope == "composite":
        f = files[files.source == "aggregated"]
        return f[["path", "source", "class", "phash", "dhash",
                  "sha256"]].reset_index(drop=True)

    raise SystemExit(f"unknown scope {scope}")


def load_grey(p: str) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(p).convert("L")
                          .resize((SSIM_SIZE, SSIM_SIZE)), dtype=np.float32)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--threshold", type=int, default=2,
                    help="dual-hash overlap threshold; keep at the value "
                         "calibrated on the development data")
    ap.add_argument("--group-threshold", type=int, default=2,
                    help="internal near-duplicate clustering threshold")
    ap.add_argument("--ssim-sample", type=int, default=400,
                    help="matched pairs to confirm with SSIM (0 to skip)")
    ap.add_argument("--published-per-class", type=int, default=None)
    ap.add_argument("--montage", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    EXTERNAL.mkdir(parents=True, exist_ok=True)
    ext = index_external(Path(a.root))
    n_raw = len(ext)

    # -- counts, stated at every stage ----------------------------------------
    print("\n" + "=" * 70)
    print("Composition as distributed")
    print("=" * 70)
    print(f"  files on disk: {n_raw}")
    print(ext.label.value_counts().to_string())
    if a.published_per_class:
        pub = a.published_per_class * ext.label.nunique()
        print(f"\n  published composition: {a.published_per_class} per class "
              f"({pub} total)")
        if n_raw != pub:
            print(f"  DISCREPANCY: {n_raw} files distributed against {pub} "
                  f"published.\n  These counts are of files as shipped, before "
                  f"any deduplication here.\n  Report both; do not substitute "
                  f"one for the other.")

    dup_exact = ext[ext.duplicated("sha256", keep=False)]
    n_exact_groups = dup_exact.sha256.nunique()
    print(f"\n  byte-identical duplicate files within the release: "
          f"{len(dup_exact)} in {n_exact_groups} groups")

    dims = ext.width.astype(str) + "x" + ext.height.astype(str)
    print(f"  distinct dimensions: {dims.nunique()} "
          f"(most common {dims.value_counts().index[0]})")

    # -- internal near-duplicate grouping --------------------------------------
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

    # -- one comparison per scope ---------------------------------------------
    results, per_scope = {}, {}
    for scope in ["trained", "sources", "composite"]:
        dev = load_scope(scope)
        if not len(dev):
            print(f"\n  scope '{scope}': no development images, skipped")
            continue
        print("\n" + "=" * 70)
        print(f"Scope '{scope}': {len(dev)} development images")
        print("=" * 70)

        bi, bd = dual_nearest(ph, dh, hex_to_packed(dev.phash),
                              hex_to_packed(dev.dhash))
        hit = bd <= a.threshold
        ext[f"{scope}_distance"] = bd
        ext[f"{scope}_match_path"] = dev.path.to_numpy()[bi]
        ext[f"{scope}_match_source"] = dev.source.to_numpy()[bi]
        ext[f"{scope}_match_class"] = dev["class"].to_numpy()[bi]
        ext[f"{scope}_overlap"] = hit

        n_exact = len(set(ext.sha256) & set(dev.sha256[dev.sha256 != ""]))
        # A group is contaminated if ANY member matches: the others are
        # re-encodings of the same scan.
        gflag = ext.groupby("group_id")[f"{scope}_overlap"].any()
        g_hit = int(gflag.sum())

        print(f"  byte-identical matches: {n_exact}")
        print(f"  images at dual-hash <= {a.threshold}: {int(hit.sum())} / "
              f"{n_raw} ({100 * hit.mean():.1f}%)")
        print(f"  GROUPS contaminated: {g_hit} / {n_groups} "
              f"({100 * g_hit / n_groups:.1f}%)")
        print(f"  minimum distance: {int(bd.min())}")

        tbl = pd.crosstab(ext.label, ext[f"{scope}_overlap"])
        tbl.columns = [str(c) for c in tbl.columns]
        if "True" in tbl:
            tbl["rate_%"] = (100 * tbl["True"] / tbl.sum(axis=1)).round(1)
        print("\n  by class:")
        print(tbl.to_string())

        if hit.any():
            print("\n  matched images by external class and development source:")
            print(pd.crosstab(ext.label[hit],
                              ext[f"{scope}_match_source"][hit]).to_string())
            print("\n  label agreement of matched pairs:")
            agree = (ext.label[hit] == ext[f"{scope}_match_class"][hit]).mean()
            print(f"    {100 * agree:.1f}% of matches share their diagnostic "
                  f"label\n    (random pairing would give roughly "
                  f"{100 / max(ext.label.nunique(), 1):.0f}%)")

        results[scope] = {
            "development_images": len(dev),
            "byte_identical": n_exact,
            "images_matched": int(hit.sum()),
            "image_rate": float(hit.mean()),
            "groups_contaminated": g_hit,
            "group_rate": float(g_hit / n_groups),
            "min_distance": int(bd.min()),
            "per_class": tbl.to_dict(),
        }
        per_scope[scope] = gflag

    # -- SSIM confirmation -----------------------------------------------------
    # The scope that matters for independence is 'sources'; confirm there.
    conf_scope = "sources" if "sources" in results else next(iter(results), None)
    if a.ssim_sample and conf_scope:
        hit_idx = np.where(ext[f"{conf_scope}_overlap"].to_numpy())[0]
        if len(hit_idx):
            rng = np.random.default_rng(a.seed)
            take = rng.choice(hit_idx, min(a.ssim_sample, len(hit_idx)),
                              replace=False)
            print("\n" + "=" * 70)
            print(f"SSIM confirmation on {len(take)} matched pairs "
                  f"(scope '{conf_scope}')")
            print("=" * 70)
            vals, dists = [], []
            for i in tqdm(take, unit="pair"):
                A = load_grey(ext.path.iloc[i])
                B = load_grey(ext[f"{conf_scope}_match_path"].iloc[i])
                if A is None or B is None:
                    continue
                vals.append(ssim(A, B, data_range=255.0))
                dists.append(int(ext[f"{conf_scope}_distance"].iloc[i]))
            v, d = np.array(vals), np.array(dists)
            print(f"  SSIM  median {np.median(v):.3f}   "
                  f"mean {v.mean():.3f}   min {v.min():.3f}")
            for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95),
                           (0.95, 1.01)]:
                m = (v >= lo) & (v < hi)
                if m.sum():
                    print(f"    SSIM [{lo:.2f}, {hi:.2f}): {int(m.sum()):4d} "
                          f"pairs, mean hash distance {d[m].mean():.1f}")
            results["ssim"] = {"scope": conf_scope, "n": int(len(v)),
                               "median": float(np.median(v)),
                               "mean": float(v.mean()),
                               "min": float(v.min()),
                               "frac_above_0.95": float((v >= 0.95).mean()),
                               "frac_above_0.90": float((v >= 0.90).mean())}
            print(f"\n  {100 * (v >= 0.95).mean():.1f}% of matched pairs have "
                  f"SSIM >= 0.95.")
            print("  Hash distance and SSIM are independent measures; agreement")
            print("  between them is what makes the overlap claim defensible.")

    # -- what a decontaminated cohort would look like --------------------------
    if "sources" in per_scope:
        keep_groups = set(per_scope["sources"][~per_scope["sources"]].index)
        kept = ext[ext.group_id.isin(keep_groups)]
        print("\n" + "=" * 70)
        print("Cohort remaining after excluding contaminated groups "
              "(scope 'sources')")
        print("=" * 70)
        summary = pd.DataFrame({
            "distributed": ext.label.value_counts(),
            "retained": kept.label.value_counts(),
        }).fillna(0).astype(int)
        summary["removed"] = summary.distributed - summary.retained
        summary["retained_%"] = (100 * summary.retained /
                                 summary.distributed).round(1)
        print(summary.to_string())
        empty = summary[summary.retained == 0]
        print()
        if len(empty):
            print(f"  Classes reduced to zero: {', '.join(empty.index)}.")
            print("  A four-class external evaluation cannot be constructed at")
            print("  any filtering level. Report the screening; do not present")
            print("  the remainder as external validation.")
        elif summary.retained.sum() < 0.5 * n_raw:
            print("  More than half the cohort is excluded. The provenance of")
            print("  what remains is unestablished rather than established:")
            print("  perceptual hashing finds detectable reuse, not all reuse.")
        else:
            print("  A decontaminated cohort is available. State the exclusion")
            print("  counts and treat it as a secondary cross-dataset analysis,")
            print("  not a fully independent external cohort.")
        results["retained_cohort"] = summary.to_dict()

    if a.montage and conf_scope:
        write_montages(ext, a.name, conf_scope, a.montage)

    ext.to_csv(EXTERNAL / f"{a.name}_manifest.csv", index=False)
    report = {"name": a.name, "root": str(a.root),
              "files_distributed": n_raw,
              "published_per_class": a.published_per_class,
              "class_counts_as_distributed": ext.label.value_counts().to_dict(),
              "exact_duplicate_files": int(len(dup_exact)),
              "exact_duplicate_groups": int(n_exact_groups),
              "near_duplicate_groups": int(n_groups),
              "overlap_threshold": a.threshold,
              "scopes": results}
    (EXTERNAL / f"{a.name}_overlap_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\n  wrote {EXTERNAL / f'{a.name}_manifest.csv'}")
    print(f"  wrote {EXTERNAL / f'{a.name}_overlap_report.json'}")


def write_montages(ext: pd.DataFrame, name: str, scope: str, n: int) -> None:
    """Closest pairs, stratified over distance, so the judgement is not made
    only on the easiest examples."""
    d = EXTERNAL / f"{name}_pairs_{scope}"
    d.mkdir(parents=True, exist_ok=True)
    T = 240
    hit = ext[ext[f"{scope}_overlap"]]
    if not len(hit):
        return
    per_bucket = max(1, n // 3)
    picks = []
    for lo, hi in [(0, 0), (1, 2), (3, 99)]:
        b = hit[(hit[f"{scope}_distance"] >= lo) &
                (hit[f"{scope}_distance"] <= hi)]
        if len(b):
            picks.append(b.sample(min(per_bucket, len(b)), random_state=0))
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