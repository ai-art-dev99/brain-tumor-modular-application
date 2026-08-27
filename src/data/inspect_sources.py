#!/usr/bin/env python3
"""
inspect_sources.py -- diagnostics on the raw source index.

Answers three questions the first manifest run raised:
  1. Which folders do the 801 unclassified BR35H images live in?
  2. Where do the 1,128 byte-identical duplicate groups actually fall:
     within a source, or across sources?
  3. Is SARTAJ's 500-image no_tumor folder a subset of BR35H's 1,500?
     If so, the 'no tumour' class of the composite dataset was already
     duplicated before any modelling began.

Usage:  python inspect_sources.py
"""

from pathlib import Path

import pandas as pd

MANIFEST = Path("/workspace/data/manifest")
df = pd.read_csv(MANIFEST / "files_index.csv")

pd.set_option("display.width", 140)

# =============================================================================
# 1. Unclassified images: which directories are they in?
# =============================================================================
print("=" * 70)
print("1. Folders holding images that did not map to a known class")
print("=" * 70)

unmapped = df[df["class"].isin(["unknown", "unlabelled", "tumour_unspecified"])]
if len(unmapped):
    # Show the directory two levels below the source root, which is where
    # these repositories keep their class folders.
    dirs = unmapped.path.map(lambda p: str(Path(p).parent))
    summary = (unmapped.assign(folder=dirs)
                       .groupby(["source", "class", "folder"])
                       .size().rename("n").reset_index()
                       .sort_values("n", ascending=False))
    print(summary.to_string(index=False))
else:
    print("  none")

# =============================================================================
# 2. Duplicate groups: within-source vs cross-source
# =============================================================================
print()
print("=" * 70)
print("2. Byte-identical duplicate groups (sha256)")
print("=" * 70)

dups = df[df.duplicated("sha256", keep=False)]
groups = dups.groupby("sha256")

rows = []
for h, g in groups:
    srcs = tuple(sorted(g.source.unique()))
    classes = tuple(sorted(g["class"].unique()))
    rows.append({
        "sha256": h,
        "n_files": len(g),
        "sources": " + ".join(srcs),
        "n_sources": len(srcs),
        "classes": " + ".join(classes),
        "class_conflict": len(classes) > 1,
    })
gdf = pd.DataFrame(rows)

print(f"  duplicate files : {len(dups)}")
print(f"  duplicate groups: {len(gdf)}")
print()
print("  groups by source combination:")
print(gdf.groupby("sources").agg(groups=("sha256", "size"),
                                 files=("n_files", "sum")).to_string())

# A duplicate group whose members carry different class labels is a labelling
# contradiction: the identical image is filed as two different diagnoses.
# These cannot both be right and must be excluded or adjudicated.
conflict = gdf[gdf.class_conflict]
print()
print(f"  groups where the SAME image carries DIFFERENT class labels: {len(conflict)}")
if len(conflict):
    print(conflict.groupby("classes").size().rename("groups").to_string())
    print("  -> these are label contradictions, not just duplicates.")
    conflict.to_csv(MANIFEST / "label_conflicts.csv", index=False)
    print(f"  -> written to {MANIFEST / 'label_conflicts.csv'}")

# =============================================================================
# 3. Is SARTAJ's no_tumor folder a subset of BR35H's?
# =============================================================================
print()
print("=" * 70)
print("3. Overlap between the two 'no tumour' sources")
print("=" * 70)

sartaj_no = set(df[(df.source == "sartaj") & (df["class"] == "notumor")].sha256)
br35h_no = set(df[(df.source == "br35h") & (df["class"] == "notumor")].sha256)

print(f"  SARTAJ notumor : {len(sartaj_no)} distinct hashes")
print(f"  BR35H  notumor : {len(br35h_no)} distinct hashes")
overlap = sartaj_no & br35h_no
print(f"  exact overlap  : {len(overlap)}")
if sartaj_no:
    print(f"  -> {100 * len(overlap) / len(sartaj_no):.1f}% of SARTAJ's "
          f"no-tumour images are byte-identical to BR35H images")
    if len(overlap) > 0.5 * len(sartaj_no):
        print("  -> the composite 'no tumour' class was duplicated at source. "
              "Every\n     model trained on it, including ours, saw the same "
              "images twice.")

# =============================================================================
# 4. Within-source duplication rate
# =============================================================================
print()
print("=" * 70)
print("4. Duplication rate within each source")
print("=" * 70)

for src, g in df.groupby("source"):
    uniq = g.sha256.nunique()
    print(f"  {src:<12} {len(g):5d} files -> {uniq:5d} unique "
          f"({100 * (1 - uniq / len(g)):.1f}% redundant)")

print()
print("Note: this counts EXACT duplicates only. Near-duplicates (re-encoded,")
print("resized, or adjacent slices of one patient) are invisible to sha256 and")
print("are handled separately in dedup.py using perceptual hashing.")