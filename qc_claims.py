#!/usr/bin/env python3
"""
qc_claims.py -- verify every number in the manuscript and response letter
against the artefacts that produced them.

WHY
---
Numbers have moved between the manuscript, the response letter and the
supplementary tables during this revision, and several were transcribed from
console output rather than read from a file. The original submission contained
at least six transcription errors of that kind. This script removes the
possibility by checking two things for every claim:

  1. the value computed from the stored artefact equals the value we intend to
     report, and
  2. that value, formatted exactly as it should appear, is present in the
     document.

A claim that drifts in either direction fails. A claim whose artefact is
missing is reported as SKIP rather than silently passing.

The final pass lists numbers that appear in the documents but are not covered
by any claim, so the unverified remainder is visible rather than assumed.

USAGE
-----
    python qc_claims.py
    python qc_claims.py --manuscript path.md --letter path.docx
    python qc_claims.py --list          # show the claim registry only
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

MANIFEST = Path("/workspace/data/manifest")
SPLITS = MANIFEST / "splits"
RUNS = Path("/workspace/outputs/runs")
PROBES = Path("/workspace/outputs/probes")
EXTERNAL = Path("/workspace/data/external")
EXT_OUT = Path("/workspace/outputs/external")
TABLES = Path("/workspace/outputs/tables")


# =============================================================================
# Artefact access, cached and fail-soft
# =============================================================================

_cache: dict = {}


class Missing(Exception):
    """Raised when an artefact required by a claim is absent."""


def _load(path: Path, loader: Callable):
    key = str(path)
    if key not in _cache:
        if not path.exists():
            raise Missing(str(path))
        _cache[key] = loader(path)
    return _cache[key]


def csv(path: Path) -> pd.DataFrame:
    return _load(path, pd.read_csv)


def js(path: Path) -> dict:
    return _load(path, lambda p: json.loads(p.read_text()))


def run_metrics(tag: str) -> dict:
    return js(RUNS / tag / "metrics.json")


def point(tag: str, model: str, key: str) -> float:
    m = run_metrics(tag)["models"]
    if model not in m:
        raise Missing(f"{tag}:{model}")
    p = m[model]
    # train_eval v2 nests under "point"; older runs are flat
    return float(p["point"][key] if "point" in p else p[key])


def ci(tag: str, model: str, key: str, bound: int) -> float:
    m = run_metrics(tag)["models"][model]
    c = m.get("ci95", {})
    if key in c:
        return float(c[key][bound])
    return float(m[f"{key}_ci95"][bound])


def comparison(tag: str, a: str, b: str, field_: str):
    for c in run_metrics(tag)["comparisons"]:
        if {c["model_a"], c["model_b"]} == {a, b}:
            v = c[field_]
            if isinstance(v, (list, tuple)):
                # interval bounds flip with the sign convention
                return [-v[1], -v[0]] if c["model_a"] != a else list(v)
            if field_ == "acc_diff" and c["model_a"] != a:
                v = -v
            return float(v)
    raise Missing(f"comparison {a} vs {b} in {tag}")


def dataset() -> pd.DataFrame:
    d = csv(MANIFEST / "dataset.csv")
    d["patient_id"] = d.patient_id.fillna("")
    d["mask_path"] = d.mask_path.fillna("")
    return d


def split(name: str) -> pd.DataFrame:
    d = csv(SPLITS / f"splits_{name}_outer.csv")
    d["patient_id"] = d.patient_id.fillna("")
    return d


def sibling_rate(name: str) -> float:
    """Share of test images having a same-group sibling in their training fold."""
    d = split(name)
    bad = tot = 0
    for k in sorted(d.outer_fold.unique()):
        te = d[d.outer_fold == k]
        tr_groups = set(d[d.outer_fold != k].group_id)
        bad += int(te.group_id.isin(tr_groups).sum())
        tot += len(te)
    return 100 * bad / tot


def fold_std(tag: str, model: str) -> float:
    pf = csv(RUNS / tag / "per_fold.csv")
    return float(pf[pf.model == model].accuracy.std())


def fold_std_range(tag: str) -> tuple[float, float]:
    pf = csv(RUNS / tag / "per_fold.csv")
    sd = pf.groupby("model").accuracy.std()
    return float(sd.min()), float(sd.max())


def cal(tag: str, model: str, key: str, level: str = "unit") -> float:
    """Calibration figures. `level` is 'image' or 'unit'; the unit level is
    labelled 'group' or 'patient' depending on the configuration, so it is
    selected by exclusion rather than by name."""
    d = csv(RUNS / tag / "calibration.csv")
    d = d[d.level != "image"] if level == "unit" else d[d.level == level]
    r = d[d.model == model]
    if not len(r):
        raise Missing(f"calibration.csv has no {model} at level {level}")
    return float(r.iloc[0][key])


def defer(tag: str, model: str, frac: float, col: str) -> float:
    d = csv(RUNS / tag / "deferral_curve.csv")
    r = d[(d.model == model) & (np.isclose(d.defer_frac, frac))]
    if not len(r):
        raise Missing(f"deferral_curve.csv has no {model} at defer={frac}")
    return float(r.iloc[0][col])


def aurc(tag: str, model: str) -> float:
    f = RUNS / tag / "risk_coverage.csv"
    if not f.exists():
        raise Missing("risk_coverage.csv not written; re-run explain.py "
                      "--mode deferral with the patched version")
    d = csv(f)
    r = d[d.model == model]
    if not len(r):
        raise Missing(f"risk_coverage.csv has no {model}")
    return float(r.iloc[0]["aurc"])


def probe(name: str, key: str) -> float:
    return float(js(PROBES / f"{name}.json")[key])


def ext_report(name: str, *path):
    """Read a value from an audit report.

    Reports written by earlier versions of audit_external use a different
    schema. Rather than silently reading the wrong field, a missing key is
    raised as Missing so the claim is reported as SKIP and the audit is
    re-run against the current version.
    """
    d = js(EXTERNAL / f"{name}_overlap_report.json")
    for k in path:
        if not isinstance(d, dict) or k not in d:
            raise Missing(f"{name}_overlap_report.json lacks {'.'.join(path)} "
                          f"(written by an older audit_external; re-run it)")
        d = d[k]
    return d


def ext_manifest(name: str) -> pd.DataFrame:
    """The audit manifest is one row per image and is schema-stable across
    versions of audit_external, unlike the summary report. Anything derivable
    from it is read from here."""
    d = csv(EXTERNAL / f"{name}_manifest.csv")
    if "published_split" not in d.columns:
        d["published_split"] = "unassigned"
    return d


def paired(mode: str, model: str, field_: str) -> float:
    d = js(EXT_OUT / f"paired_{mode}_all" / "summary.json")
    for r in d["results"]:
        if r["model"] == model:
            return float(r[field_])
    raise Missing(f"paired {mode}:{model}")


def occlusion(tag: str, label: str, correct: bool, col: str) -> float:
    d = csv(RUNS / tag / "occlusion_svm.csv")
    sub = d[(d.label == label) & (d.correct.astype(bool) == correct)]
    if not len(sub):
        raise Missing(f"occlusion {tag}:{label}")
    return float(sub[col].mean())


# =============================================================================
# Formatting
# =============================================================================

def distinct_dims(source: str) -> int:
    d = csv(MANIFEST / "files_index.csv")
    d = d[d.source == source]
    return (d.width.astype(str) + "x" + d.height.astype(str)).nunique()


def retained_after(name: str, scope: str) -> pd.DataFrame:
    """Images whose near-duplicate group contains no match in `scope`."""
    d = ext_manifest(name)
    flag = d.groupby("group_id")[f"{scope}_overlap"].transform("any")
    return d[~flag]


RAW_BR35H = Path("/workspace/data/raw/br35h")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(root: Path, subdir: str) -> int:
    d = root / subdir
    if not d.exists():
        for cand in root.rglob(subdir):
            if cand.is_dir():
                d = cand
                break
        else:
            raise Missing(f"{root}/{subdir} not found")
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def cross_source_clusters() -> int:
    d = csv(MANIFEST / "dataset_with_duplicates.csv")
    per = d.groupby("dup_cluster").source.nunique()
    return int((per > 1).sum())


def sartaj_notumor() -> dict:
    f = csv(MANIFEST / "files_index.csv")
    sn = f[(f.source == "sartaj") & (f["class"] == "notumor")]
    br = set(f[(f.source == "br35h")].sha256)
    uniq = sn.sha256.nunique()
    return {"n": len(sn), "unique": uniq,
            "dup_pct": 100 * (1 - uniq / max(len(sn), 1)),
            "in_br35h": int(sn.sha256.isin(br).sum())}


def pub_leak() -> dict:
    """Leakage in the partition distributed with the benchmark, recomputed
    from the provenance manifest rather than quoted from console output."""
    m = csv(MANIFEST / "manifest.csv")
    r = m[m.best_distance <= 4]
    # matched_class is not stored in the manifest, so the recovered label is
    # joined back from the Figshare index rather than left as NaN, which an
    # earlier version reported as a passing claim.
    if "matched_class" in r.columns:
        agree = 100 * (r["class"] == r.matched_class).mean()
    elif "matched_mat" in r.columns:
        fig = csv(MANIFEST / "figshare_index.csv").set_index("mat_file")["class"]
        agree = 100 * (r["class"] == r.matched_mat.map(fig)).mean()
    else:
        raise Missing("manifest.csv records neither matched_class nor matched_mat")
    withp = r[r.patient_id.fillna("") != ""]
    spread = withp.groupby("patient_id").orig_split.nunique()
    bad = spread[spread > 1].index
    return {"matched": len(r), "agreement": agree,
            "patients": withp.patient_id.nunique(),
            "spanning": len(bad),
            "images": int(withp.patient_id.isin(bad).sum())}


def strata() -> dict:
    d = csv(EXT_OUT / "bdneuro_v7_manifest_test" / "predictions.csv") \
        if (EXT_OUT / "bdneuro_v7_manifest_test" / "predictions.csv").exists() \
        else None
    if d is None:
        m = ext_manifest("bdneuro_v7")
        m = m[m.published_split == "test"]
        m = m[m.label.isin(["glioma", "meningioma", "notumor", "pituitary"])]
        direct = m.trained_overlap.astype(bool)
        src = m.sources_overlap.astype(bool)
        clean = ~src
        strict = clean & (m.sources_distance >= 12)
        return {"direct": int(direct.sum()),
                "source_only": int((src & ~direct).sum()),
                "clean": int(clean.sum()), "strict": int(strict.sum()),
                "strict_pct": 100 * strict.sum() / max(int(clean.sum()), 1)}
    c = d.stratum.value_counts()
    strict = int((d.sources_distance >= 12).sum() and
                 ((d.stratum == "clean") & (d.sources_distance >= 12)).sum())
    return {"direct": int(c.get("direct", 0)),
            "source_only": int(c.get("source_only", 0)),
            "clean": int(c.get("clean", 0)), "strict": strict,
            "strict_pct": 100 * strict / max(int(c.get("clean", 1)), 1)}


def f4(v):    return f"{v:.4f}"
def f3(v):    return f"{v:.3f}"
def f2(v):    return f"{v:.2f}"
def f1(v):    return f"{v:.1f}"
def thou(v):  return f"{int(round(v)):,}"
def i(v):     return str(int(round(v)))
def pp2(v):   return f"{v:+.2f}"          # percentage-point difference
def pct1(v):  return f"{v:.1f}"


@dataclass
class Claim:
    id: str
    where: str                   # section of the manuscript
    truth: Callable[[], float]   # computed from the artefact
    fmt: Callable                # how it must appear
    note: str = ""
    docs: tuple = ("manuscript", "letter")
    # Some claims exist to pin an artefact value that the documents do not
    # quote. They are checked for resolvability and recorded, but absence from
    # the text is not a failure.
    must_appear: bool = True
    # Set where zero is a real, expected value rather than a sign that the
    # artefact has lost the category.
    zero_ok: bool = False
    aliases: list = field(default_factory=list)   # other renderings that count


# =============================================================================
# Claim registry
# =============================================================================

def build_claims() -> list[Claim]:
    C = []
    add = C.append

    # -- dataset construction -------------------------------------------------
    add(Claim("ds.figshare_slices", "3.1",
              lambda: len(csv(MANIFEST / "figshare_index.csv")), thou))
    add(Claim("ds.figshare_patients", "3.1",
              lambda: csv(MANIFEST / "figshare_index.csv").patient_id.nunique(), i))
    add(Claim("ds.slices_per_patient_mean", "3.1",
              lambda: csv(MANIFEST / "figshare_index.csv")
                      .groupby("patient_id").size().mean(), f1))
    add(Claim("ds.retained_total", "3.3", lambda: len(dataset()), thou))
    add(Claim("ds.groups_total", "3.3",
              lambda: dataset().group_id.nunique(), i))
    add(Claim("ds.mask_count", "3.3",
              lambda: int((dataset().mask_path != "").sum()), thou))
    add(Claim("ds.mask_pct", "3.3",
              lambda: 100 * (dataset().mask_path != "").mean(), pct1))
    add(Claim("ds.sartaj_survivors", "3.3",
              lambda: int((dataset().source == "sartaj").sum()), i))

    for cls, sec in [("glioma", "3.3"), ("meningioma", "3.3"),
                     ("pituitary", "3.3"), ("notumor", "3.3")]:
        add(Claim(f"ds.retained.{cls}", sec,
                  lambda c=cls: int((dataset().label == c).sum()), thou))
        add(Claim(f"ds.groups.{cls}", sec,
                  lambda c=cls: dataset()[dataset().label == c]
                                .group_id.nunique(), i))
        add(Claim(f"ds.per_group.{cls}", sec,
                  lambda c=cls: (dataset().label == c).sum() /
                                dataset()[dataset().label == c].group_id.nunique(),
                  f1))

    # -- partitioning ---------------------------------------------------------
    add(Claim("split.imagelevel.sibling_pct", "3.4",
              lambda: sibling_rate("main_imagelevel"), pct1))
    add(Claim("split.imagelevel.patients_spanning", "3.4",
              lambda: int((split("main_imagelevel")
                           .query("patient_id != ''")
                           .groupby("patient_id").outer_fold.nunique() > 1).sum()), i))
    add(Claim("split.grouped.sibling_pct", "3.4",
              lambda: sibling_rate("main"), lambda v: f"{v:.1f}",
              note="must be 0.0", zero_ok=True))

    # -- headline performance -------------------------------------------------
    for tag, models in [("main_finetuned_v2",
                         ["cnn", "cnn_svm", "cnn_mlp", "cnn_logreg",
                          "cnn_rf", "cnn_knn"]),
                        ("figshare_finetuned_v2",
                         ["cnn", "cnn_svm", "cnn_mlp", "cnn_logreg",
                          "cnn_rf", "cnn_knn"]),
                        ("main_naive_ft_v2",
                         ["cnn", "cnn_svm", "cnn_mlp", "cnn_logreg",
                          "cnn_rf", "cnn_knn"])]:
        for m in models:
            add(Claim(f"perf.{tag}.{m}.acc", "4.1",
                      lambda t=tag, mm=m: point(t, mm, "accuracy"), f4))
            add(Claim(f"perf.{tag}.{m}.bal", "4.1",
                      lambda t=tag, mm=m: point(t, mm, "balanced_accuracy"), f4,
                      # Table 7 reports accuracy only for the image-level run,
                      # so its balanced figures are pinned rather than quoted.
                      must_appear=(tag != "main_naive_ft_v2")))

    for m in ["cnn", "cnn_svm"]:
        for b, lbl in [(0, "lo"), (1, "hi")]:
            add(Claim(f"perf.main.{m}.ci_{lbl}", "4.1",
                      lambda mm=m, bb=b: ci("main_finetuned_v2", mm,
                                            "accuracy", bb), f4))
            add(Claim(f"perf.fig.{m}.ci_{lbl}", "4.1",
                      lambda mm=m, bb=b: ci("figshare_finetuned_v2", mm,
                                            "accuracy", bb), f4))

    # -- inflation ------------------------------------------------------------
    for m in ["cnn", "cnn_svm", "cnn_knn", "cnn_mlp", "cnn_logreg", "cnn_rf"]:
        add(Claim(f"infl.{m}", "4.2",
                  lambda mm=m: 100 * (point("main_naive_ft_v2", mm, "accuracy")
                                      - point("main_finetuned_v2", mm, "accuracy")),
                  f2))

    # The manuscript states a range across model configurations, so the claim
    # must target the range rather than one model.
    for tag, lbl in [("main_finetuned_v2", "grouped"),
                     ("main_naive_ft_v2", "naive")]:
        add(Claim(f"infl.foldstd.{lbl}.min", "4.2",
                  lambda t=tag: fold_std_range(t)[0], f3))
        add(Claim(f"infl.foldstd.{lbl}.max", "4.2",
                  lambda t=tag: fold_std_range(t)[1], f3))

    # -- primary comparison ---------------------------------------------------
    add(Claim("cmp.main.diff_pp", "4.3",
              lambda: 100 * abs(comparison("main_finetuned_v2",
                                           "cnn", "cnn_svm", "acc_diff")), f2))
    add(Claim("cmp.main.p", "4.3",
              lambda: comparison("main_finetuned_v2", "cnn", "cnn_svm",
                                 "bootstrap_p"), f3))
    add(Claim("cmp.fig.diff_pp", "4.3",
              lambda: 100 * abs(comparison("figshare_finetuned_v2",
                                           "cnn", "cnn_svm", "acc_diff")), f2))
    add(Claim("cmp.fig.p", "4.3",
              lambda: comparison("figshare_finetuned_v2", "cnn", "cnn_svm",
                                 "bootstrap_p"), f3))
    add(Claim("cmp.main.bal_diff_pp", "4.3",
              lambda: 100 * (point("main_finetuned_v2", "cnn_svm",
                                   "balanced_accuracy")
                             - point("main_finetuned_v2", "cnn",
                                     "balanced_accuracy")), f2,
              note="reported alongside the accuracy difference"))

    # -- per-class results, Table 8 ------------------------------------------
    for m in ["cnn", "cnn_svm"]:
        for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
            add(Claim(f"cls.{m}.{cls}.recall", "4.3",
                      lambda mm=m, c=cls: point("main_finetuned_v2", mm,
                                                f"recall::{c}"), f3))
            add(Claim(f"cls.{m}.{cls}.f1", "4.3",
                      lambda mm=m, c=cls: point("main_finetuned_v2", mm,
                                                f"f1::{c}"), f3))
            for b, lbl in [(0, "lo"), (1, "hi")]:
                add(Claim(f"cls.{m}.{cls}.recall_{lbl}", "4.3",
                          lambda mm=m, c=cls, bb=b: ci("main_finetuned_v2", mm,
                                                       f"recall::{c}", bb), f3))

    # -- source probes --------------------------------------------------------
    add(Claim("probe.bg_pair.bal", "4.4",
              lambda: probe("main_background_figshare-br35h",
                            "balanced_accuracy"), f4))
    add(Claim("probe.bg_meningioma.bal", "4.4",
              lambda: probe("main_background_meningioma",
                            "balanced_accuracy"), f4))
    add(Claim("probe.full.bal", "4.4",
              lambda: probe("main_full", "balanced_accuracy"), f4))
    add(Claim("probe.brain_crop.bal", "4.4",
              lambda: probe("main_brain_crop", "balanced_accuracy"), f4))

    # -- attribution ----------------------------------------------------------
    for cls in ["glioma", "meningioma", "pituitary"]:
        add(Claim(f"occ.fig.{cls}.correct", "4.5",
                  lambda c=cls: occlusion("figshare_finetuned_v2", c, True,
                                          "concentration"), f2))
        add(Claim(f"occ.fig.{cls}.incorrect", "4.5",
                  lambda c=cls: occlusion("figshare_finetuned_v2", c, False,
                                          "concentration"), f2))

    # -- external screening ---------------------------------------------------
    # PMRAM: the three scopes answer different questions and must not be
    # pooled. An earlier audit compared against a single pool that included
    # the composite redistribution, which inflated the apparent overlap with
    # the source repositories from 33.8% to 56.4%.
    add(Claim("ext.pmram.files", "4.7",
              lambda: ext_report("pmram", "files"), thou,
              note="files present in the downloaded archive"))
    add(Claim("ext.pmram.sources_matched", "4.7",
              lambda: ext_report("pmram", "scopes", "sources",
                                 "images_matched"), i))
    add(Claim("ext.pmram.sources_pct", "4.7",
              lambda: 100 * ext_report("pmram", "scopes", "sources",
                                       "image_rate"), pct1))
    add(Claim("ext.pmram.composite_matched", "4.7",
              lambda: ext_report("pmram", "scopes", "composite",
                                 "images_matched"), i))
    add(Claim("ext.pmram.composite_pct", "4.7",
              lambda: 100 * ext_report("pmram", "scopes", "composite",
                                       "image_rate"), pct1))
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        add(Claim(f"ext.pmram.dist.{cls}", "4.7",
                  lambda c=cls: int((ext_manifest("pmram").label == c).sum()), i))
    add(Claim("ext.pmram.exact_dupes", "4.7",
              lambda: int(ext_manifest("pmram")
                          .duplicated("sha256", keep=False).sum()), i))
    add(Claim("ext.pmram.dupe_groups", "4.7",
              lambda: int(ext_manifest("pmram")
                          .pipe(lambda d: d[d.duplicated("sha256", keep=False)])
                          .sha256.nunique()), i))
    # retained counts after excluding contaminated groups, scope 'sources'
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        add(Claim(f"ext.pmram.retained.{cls}", "4.7",
                  lambda c=cls: int(retained_after(
                      "pmram", "sources").query("label == @c").shape[0]), i))
    add(Claim("ext.pmram.notumor_pct", "4.7",
              lambda: 100 * ext_manifest("pmram").query(
                  "label == 'notumor'").sources_overlap.mean(), pct1))
    add(Claim("ext.pmram.meningioma_pct", "4.7",
              lambda: 100 * ext_manifest("pmram").query(
                  "label == 'meningioma'").sources_overlap.mean(), pct1))

    # -- dataset construction, exclusions and provenance ---------------------
    add(Claim("ds.curated_total", "3.2",
              lambda: len(csv(MANIFEST / "dataset_with_duplicates.csv")), thou,
              note="rows surviving the inclusion rules, before deduplication"))
    for cls in ["glioma", "meningioma", "pituitary", "notumor"]:
        add(Claim(f"ds.curated.{cls}", "3.2",
                  lambda c=cls: int((csv(MANIFEST / "dataset_with_duplicates.csv")
                                     .label == c).sum()), thou))
    add(Claim("ds.pid_before_propagation", "3.3",
              lambda: 100 * len(csv(MANIFEST / "figshare_index.csv")) /
                      len(csv(MANIFEST / "dataset_with_duplicates.csv")), pct1))
    add(Claim("ds.pid_after_propagation", "3.3",
              lambda: 100 * (csv(MANIFEST / "dataset_with_duplicates.csv")
                             .patient_id.fillna("") != "").mean(), pct1))
    add(Claim("ds.cross_source_clusters", "3.3",
              lambda: cross_source_clusters(), thou))
    for src, cls, sec in [("figshare", "glioma", "3.1"),
                          ("figshare", "meningioma", "3.1"),
                          ("figshare", "pituitary", "3.1")]:
        add(Claim(f"ds.figshare.{cls}", sec,
                  lambda c=cls: int((csv(MANIFEST / "figshare_index.csv")["class"]
                                     == c).sum()), thou))
    for src, cls, key in [("sartaj", "glioma", "excl.sartaj_glioma"),
                          ("sartaj", "notumor", "excl.sartaj_notumor"),
                          ("br35h", "tumour_unspecified", "excl.br35h_yes")]:
        add(Claim(key, "3.2",
                  lambda s_=src, c=cls: int(((csv(MANIFEST / "files_index.csv").source == s_) &
                                             (csv(MANIFEST / "files_index.csv")["class"] == c)).sum()),
                  thou))
    # The Mask-RCNN and pred directories are excluded at indexing time, so the
    # index cannot report them. They are counted on disk instead.
    add(Claim("excl.br35h_maskrcnn", "3.2",
              lambda: count_images(RAW_BR35H, "Br35H-Mask-RCNN"), thou))
    add(Claim("excl.br35h_pred", "3.2",
              lambda: count_images(RAW_BR35H, "pred"), thou))
    # SARTAJ no-tumour redundancy, quoted in the exclusion table
    add(Claim("excl.sartaj_notumor_unique", "3.2",
              lambda: sartaj_notumor()["unique"], i))
    add(Claim("excl.sartaj_notumor_dup_pct", "3.2",
              lambda: sartaj_notumor()["dup_pct"], pct1))
    add(Claim("excl.sartaj_notumor_in_br35h", "3.2",
              lambda: sartaj_notumor()["in_br35h"], i))
    for src in ["sartaj", "br35h"]:
        add(Claim(f"ds.raw.{src}", "3.1",
                  lambda s_=src: int((csv(MANIFEST / "files_index.csv").source == s_).sum()),
                  thou))
        add(Claim(f"ds.dims.{src}", "3.1",
                  lambda s_=src: distinct_dims(s_), i))
    add(Claim("ds.sartaj_meningioma_survivors", "3.3",
              lambda: int(((dataset().source == "sartaj") &
                           (dataset().label == "meningioma")).sum()), i))

    # -- leakage in the published redistribution -----------------------------
    add(Claim("pub.matched", "3.4", lambda: pub_leak()["matched"], thou))
    add(Claim("pub.agreement", "3.4", lambda: pub_leak()["agreement"], pct1))
    add(Claim("pub.patients", "3.4", lambda: pub_leak()["patients"], i))
    add(Claim("pub.spanning", "3.4", lambda: pub_leak()["spanning"], i))
    add(Claim("pub.spanning_images", "3.4", lambda: pub_leak()["images"], thou))

    # -- external strata ------------------------------------------------------
    add(Claim("ext.strata.direct", "4.8", lambda: strata()["direct"], i))
    add(Claim("ext.strata.source_only", "4.8", lambda: strata()["source_only"], i))
    add(Claim("ext.strata.clean", "4.8", lambda: strata()["clean"], i))
    add(Claim("ext.strata.strict_clean", "4.8", lambda: strata()["strict"], i))
    add(Claim("ext.strata.strict_pct", "4.8", lambda: strata()["strict_pct"], pct1))
    add(Claim("ext.bdneuro.files", "4.7",
              lambda: ext_report("bdneuro_v7", "files"), thou))
    add(Claim("ext.bdneuro.study_images", "4.7",
              lambda: ext_report("bdneuro_v7", "files")
                      - sum(v for k, v in ext_report(
                          "bdneuro_v7", "class_by_split").get("unassigned", {}).items()),
              thou, note="files minus images outside any class directory"))
    add(Claim("ext.bdneuro.trained_matched", "4.7",
              lambda: ext_report("bdneuro_v7", "scopes", "trained",
                                 "images_matched"), thou))
    add(Claim("ext.bdneuro.sources_matched", "4.7",
              lambda: ext_report("bdneuro_v7", "scopes", "sources",
                                 "images_matched"), thou))
    add(Claim("ext.bdneuro.sources_bytes", "4.7",
              lambda: ext_report("bdneuro_v7", "scopes", "sources",
                                 "byte_identical"), thou))
    add(Claim("ext.bdneuro.composite_matched", "4.7",
              lambda: ext_report("bdneuro_v7", "scopes", "composite",
                                 "images_matched"), thou))
    add(Claim("ext.bdneuro.composite_bytes", "4.7",
              lambda: ext_report("bdneuro_v7", "scopes", "composite",
                                 "byte_identical"), thou))
    add(Claim("ext.bdneuro.trained_bytes", "4.7",
              lambda: ext_report("bdneuro_v7", "scopes", "trained",
                                 "byte_identical"), i))

    # -- calibration and selective prediction ---------------------------------
    for m, sec in [("cnn", "4.6"), ("cnn_svm", "4.6"),
                   ("cnn_logreg", "4.6"), ("cnn_rf", "4.6")]:
        add(Claim(f"cal.main.{m}.ece", sec,
                  lambda mm=m: cal("main_finetuned_v2", mm, "ece"), f3))
    for m in ["cnn", "cnn_svm", "cnn_logreg"]:
        add(Claim(f"aurc.main.{m}", "4.6",
                  lambda mm=m: aurc("main_finetuned_v2", mm), f4))
    # aggregation to the decision unit, with no deferral
    add(Claim("defer.main.cnn.unit_acc0", "4.6",
              lambda: defer("main_finetuned_v2", "cnn", 0.0,
                            "unit_accuracy"), f4))
    # the operating point quoted in the text
    add(Claim("defer.fig.cnn_svm.acc0", "4.6",
              lambda: defer("figshare_finetuned_v2", "cnn_svm", 0.0,
                            "unit_accuracy"), f3))
    add(Claim("defer.fig.cnn_svm.acc20", "4.6",
              lambda: defer("figshare_finetuned_v2", "cnn_svm", 0.20,
                            "unit_accuracy"), f3))
    add(Claim("defer.fig.images_transferred", "4.6",
              lambda: 3038 - defer("figshare_finetuned_v2", "cnn_svm", 0.20,
                                   "images_kept"), i))

    # -- external label agreement and structural similarity ------------------
    for name, tag in [("pmram", "ext.pmram"), ("bdneuro_v7", "ext.bdneuro")]:
        add(Claim(f"{tag}.label_agreement", "4.7",
                  lambda n=name: 100 * ext_manifest(n)
                      .query("sources_overlap")
                      .pipe(lambda d: (d.label == d.sources_match_class).mean()),
                  lambda v: f"{v:.1f}"))
        add(Claim(f"{tag}.ssim_median", "4.7",
                  lambda n=name: ext_report(n, "scopes", "ssim", "median"), f3))
        add(Claim(f"{tag}.ssim_frac95", "4.7",
                  lambda n=name: 100 * ext_report(n, "scopes", "ssim",
                                                  "frac_above_0.95"), pct1))
    add(Claim("ext.bdneuro.sources_pct", "4.7",
              lambda: 100 * ext_report("bdneuro_v7", "scopes", "sources",
                                       "image_rate"), pct1))
    add(Claim("ext.bdneuro.composite_pct", "4.7",
              lambda: 100 * ext_report("bdneuro_v7", "scopes", "composite",
                                       "image_rate"), pct1))
    add(Claim("ext.bdneuro.trained_pct", "4.7",
              lambda: 100 * ext_report("bdneuro_v7", "scopes", "trained",
                                       "image_rate"), pct1))

    # -- comparison interval bounds -------------------------------------------
    for tag, lbl in [("main_finetuned_v2", "main"),
                     ("figshare_finetuned_v2", "fig")]:
        for b, side in [(0, "lo"), (1, "hi")]:
            add(Claim(f"cmp.{lbl}.ci_{side}_pp", "4.3",
                      lambda t=tag, bb=b: 100 * abs(
                          comparison(t, "cnn", "cnn_svm", "acc_diff_ci95")[bb]),
                      f2))

    # -- paired leakage contrast ---------------------------------------------
    add(Claim("paired.n", "4.8",
              lambda: js(EXT_OUT / "paired_leakage_all" /
                         "summary.json")["n_images"], thou))
    add(Claim("control.n", "4.8",
              lambda: js(EXT_OUT / "paired_control_all" /
                         "summary.json")["n_images"], thou))
    for m in ["cnn", "cnn_svm", "cnn_logreg"]:
        add(Claim(f"paired.{m}.diff", "4.8",
                  lambda mm=m: paired("leakage", mm, "diff_pp"), pp2))
        add(Claim(f"paired.{m}.lo", "4.8",
                  lambda mm=m: paired("leakage", mm, "ci_lo_pp"), f2))
        add(Claim(f"paired.{m}.hi", "4.8",
                  lambda mm=m: paired("leakage", mm, "ci_hi_pp"), f2))
        add(Claim(f"control.{m}.diff", "4.8",
                  lambda mm=m: paired("control", mm, "diff_pp"), pp2))

    return C


# =============================================================================
# Document handling
# =============================================================================

def read_docx(path: Path) -> str:
    """
    Extract text from a .docx without external tools.

    A .docx is a zip archive whose text lives in word/document.xml. Word
    frequently splits a single number across several runs -- 0.94 in one and
    05 in the next -- so runs within a paragraph are concatenated with no
    separator and only paragraphs are separated. Joining runs with a space
    instead would make every such number unfindable, and the check would pass
    documents it had never actually read.

    Table cells are ordinary paragraphs in the XML, so table contents are
    included.
    """
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = [n for n in ("word/document.xml",) if n in z.namelist()]
        if not names:
            raise Missing(f"{path} contains no word/document.xml")
        xml = z.read(names[0]).decode("utf-8", "replace")

    paragraphs = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        runs = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", para, flags=re.S)
        if not runs:
            continue
        text = "".join(runs)
        for a, b in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                     ("&quot;", '"'), ("&apos;", "'")]:
            text = text.replace(a, b)
        paragraphs.append(text)
    return "\n".join(paragraphs)


def read_doc(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".docx":
        try:
            return read_docx(path)
        except Exception as e:
            # pandoc is a fallback only; it is not required.
            try:
                return subprocess.run(["pandoc", "-t", "plain", str(path)],
                                      capture_output=True, text=True,
                                      check=True).stdout
            except Exception:
                print(f"  could not read {path}: {e}")
                return ""
    return path.read_text(encoding="utf-8")


def normalise(t: str) -> str:
    """Make hyphen and minus variants comparable, and collapse whitespace."""
    for a, b in [("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-"),
                 ("\u00a0", " ")]:
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t)


def present(needle: str, hay: str) -> bool:
    """
    Substring matching is not sufficient: 0.968 occurs inside 0.9686, so a
    claim can pass against a different number elsewhere in the document. One
    real error survived an earlier run that way. The match must therefore not
    be preceded or followed by another digit, and must not continue into a
    further decimal place.
    """
    n = normalise(needle)
    variants = [n] + ([n[1:]] if n.startswith("+") else [])
    for v in variants:
        if not v:
            continue
        pat = r"(?<![\d.,])" + re.escape(v) + r"(?![\d])"
        if re.search(pat, hay):
            return True
    return False


# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manuscript", default="manuscript_revised.md")
    ap.add_argument("--letter", default="Response_to_Editor.docx")
    ap.add_argument("--coverage", action="store_true",
                    help="list document numbers not covered by any claim")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    claims = build_claims()
    if a.list:
        for c in claims:
            print(f"  {c.id:<38} §{c.where}")
        print(f"\n  {len(claims)} claims")
        return

    docs = {"manuscript": normalise(read_doc(Path(a.manuscript))),
            "letter": normalise(read_doc(Path(a.letter)))}
    for k, v in docs.items():
        print(f"  {k}: {'loaded, ' + str(len(v)) + ' chars' if v else 'NOT FOUND'}")

    print("\n" + "=" * 78)
    print("Claim verification")
    print("=" * 78)

    ok = fail = skip = 0
    failures, covered = [], set()

    for c in claims:
        try:
            value = c.truth()
        except Missing as e:
            print(f"  SKIP  {c.id:<38} artefact missing: {e}")
            skip += 1
            continue
        except Exception as e:
            print(f"  ERROR {c.id:<38} {type(e).__name__}: {e}")
            fail += 1
            failures.append((c, "resolver error", str(e)))
            continue

        # A NaN, a zero count, or a one-character rendering can match almost
        # any document by accident. Two such claims passed silently in an
        # earlier run: an exclusion count that had become 0 because the
        # artefact no longer contained the category, and a label-agreement
        # figure that resolved to NaN. Both are now refused.
        try:
            if value != value:                       # NaN
                raise Missing("resolved to NaN")
            if isinstance(value, (int, float)) and float(value) == 0.0 \
                    and not c.zero_ok:
                raise Missing("resolved to 0, which usually means the artefact "
                              "no longer contains this category")
        except Missing as e:
            print(f"  SKIP  {c.id:<38} {e}")
            skip += 1
            continue

        rendered = c.fmt(value)
        if len(rendered.strip("+-")) < 2:
            print(f"  SKIP  {c.id:<38} rendering {rendered!r} is too short to "
                  f"match reliably")
            skip += 1
            continue
        covered.add(normalise(rendered))
        covered.add(normalise(rendered.lstrip("+")))

        if not c.must_appear:
            print(f"  note  {c.id:<38} {rendered}  (pinned, not quoted)")
            ok += 1
            continue

        where = [d for d in c.docs if docs.get(d)]
        found = {d: present(rendered, docs[d]) for d in where}

        if all(found.values()):
            print(f"  ok    {c.id:<38} {rendered}")
            ok += 1
        else:
            absent = [d for d, v in found.items() if not v]
            # absence from the letter alone is a warning: the letter quotes a
            # subset of the manuscript's numbers by design
            if absent == ["letter"]:
                print(f"  ok*   {c.id:<38} {rendered}  (not quoted in letter)")
                ok += 1
            else:
                print(f"  FAIL  {c.id:<38} artefact says {rendered}, "
                      f"absent from: {', '.join(absent)}")
                fail += 1
                failures.append((c, rendered, ", ".join(absent)))

    print("\n" + "=" * 78)
    print(f"  {ok} ok, {fail} failed, {skip} skipped")
    print("=" * 78)

    if failures:
        print("\n  Failures in detail:\n")
        for c, rendered, where in failures:
            print(f"    {c.id}  (§{c.where})")
            print(f"      artefact value : {rendered}")
            print(f"      missing from   : {where}")
            if c.note:
                print(f"      note           : {c.note}")
            print()
        print("  A failure means the document and the artefact disagree, or the")
        print("  value is written in a different format. Fix the document, or")
        print("  the claim's formatter if the document is right.")

    # -- coverage -------------------------------------------------------------
    if a.coverage and docs["manuscript"]:
        print("\n" + "=" * 78)
        print("Numbers in the manuscript not covered by any claim")
        print("=" * 78)
        nums = re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])",
                          docs["manuscript"])
        from collections import Counter
        uncovered = Counter(n for n in nums if n not in covered
                            and (("." in n) or ("," in n) or len(n) >= 3))
        for n, k in uncovered.most_common(60):
            print(f"    {n:<12} x{k}")
        print(f"\n  {len(uncovered)} distinct uncovered values. Each is either a")
        print("  reference number, a year, a hyperparameter, or a result that")
        print("  needs a claim adding to this registry before submission.")

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()