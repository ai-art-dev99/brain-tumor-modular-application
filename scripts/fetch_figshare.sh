#!/usr/bin/env bash
# =============================================================================
# fetch_figshare.sh -- download and verify the Cheng brain tumour dataset.
#
#   Article : https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
#   Content : 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients
#             stored as MATLAB v7.3 (HDF5) files.
#
# WHY THE ORIGINAL .mat FILES AND NOT A KAGGLE MIRROR
# ---------------------------------------------------
# Each .mat file carries, alongside the image:
#     cjdata.PID        -> patient identifier   (reviewer point 3: patient-level splitting)
#     cjdata.label      -> 1=meningioma 2=glioma 3=pituitary
#     cjdata.tumorMask  -> binary tumour mask   (reviewer point 9: quantitative interpretability)
#     cjdata.tumorBorder-> boundary coordinates
# Every JPEG/PNG redistribution of this dataset discards all four. Without PID
# there is no way to demonstrate patient-level splitting; without tumorMask
# there is no way to compute a pointing-game or IoU score against the model's
# saliency maps. These files are therefore the backbone of the revision.
#
# Usage:  bash fetch_figshare.sh
# =============================================================================
set -euo pipefail

ARTICLE_ID=1512427
DEST=/workspace/data/raw/figshare
API="https://api.figshare.com/v2/articles/${ARTICLE_ID}/files"

mkdir -p "${DEST}/zips" "${DEST}/mat"
cd "${DEST}/zips"

echo "==> Querying the Figshare API for the file manifest"
# Resolving file IDs from the API rather than hard-coding URLs: the IDs are
# version-specific and hard-coded links silently rot.
curl -sSL "${API}" -o files.json

python3 - <<'PY'
import json, sys
with open('files.json') as f:
    files = json.load(f)
if not files:
    sys.exit("ERROR: empty manifest. Check the article ID and network access.")
print(f"  {len(files)} file(s) listed:")
total = 0
for f in files:
    mb = f['size'] / 1e6
    total += mb
    print(f"    {f['name']:<45} {mb:8.1f} MB  md5={f['supplied_md5']}")
print(f"  total: {total:.1f} MB")
PY

echo
echo "==> Downloading"
python3 - <<'PY' > urls.txt
import json
for f in json.load(open('files.json')):
    print(f"{f['download_url']}\t{f['name']}\t{f['supplied_md5']}")
PY

while IFS=$'\t' read -r url name md5; do
    if [ -f "${name}" ]; then
        echo "  ${name}: already present, skipping download"
    else
        echo "  ${name}: downloading"
        # -L follows the redirect Figshare issues to its storage backend.
        # -C - resumes a partial file if the connection dropped earlier.
        curl -L -C - --retry 3 --retry-delay 5 -o "${name}" "${url}"
    fi

    echo -n "  ${name}: verifying md5 ... "
    actual=$(md5sum "${name}" | cut -d' ' -f1)
    if [ "${actual}" = "${md5}" ]; then
        echo "ok"
    else
        echo "MISMATCH"
        echo "    expected ${md5}"
        echo "    actual   ${actual}"
        echo "    The download is corrupt. Delete ${name} and re-run."
        exit 1
    fi
done < urls.txt

echo
echo "==> Extracting .mat files"
for z in *.zip; do
    echo "  ${z}"
    unzip -q -o "${z}" -d "${DEST}/mat"
done

# The archives may nest the files one directory deep; flatten so that later
# scripts can rely on a single glob pattern.
find "${DEST}/mat" -mindepth 2 -name '*.mat' -exec mv -t "${DEST}/mat" {} + 2>/dev/null || true
find "${DEST}/mat" -mindepth 1 -type d -empty -delete 2>/dev/null || true

COUNT=$(find "${DEST}/mat" -maxdepth 1 -name '*.mat' | wc -l)
echo
echo "==> Extracted ${COUNT} .mat files"
if [ "${COUNT}" -ne 3064 ]; then
    echo "    WARNING: expected 3064. A missing archive or a partial extraction"
    echo "    would bias every downstream patient count, so resolve this before"
    echo "    building the manifest."
fi

echo
echo "==> Structural check on the first file"
source /workspace/venv/bin/activate 2>/dev/null || true
python3 - <<'PY'
import glob, h5py, numpy as np

files = sorted(glob.glob('/workspace/data/raw/figshare/mat/*.mat'))
if not files:
    raise SystemExit("ERROR: no .mat files found after extraction.")

with h5py.File(files[0], 'r') as h:
    d = h['cjdata']
    print("  keys      :", list(d.keys()))
    # PID is a MATLAB char array; h5py returns uint16 codepoints.
    pid = ''.join(chr(c) for c in np.array(d['PID']).ravel())
    print("  PID       :", pid)
    print("  label     :", int(np.array(d['label']).ravel()[0]),
          " (1=meningioma 2=glioma 3=pituitary)")
    # h5py returns HDF5 arrays transposed relative to MATLAB's column-major order.
    img  = np.array(d['image']).T
    mask = np.array(d['tumorMask']).T
    print("  image     :", img.shape, img.dtype,
          f"range [{img.min()}, {img.max()}]")
    print("  tumorMask :", mask.shape, mask.dtype,
          f"positive px {int(mask.sum())}")

print()
print("  If PID and tumorMask are present above, these are the original files.")
PY

echo
echo "============================================================"
echo "Figshare data ready at ${DEST}/mat"
echo "Next: build_manifest.py will recover PID/label/mask for all"
echo "${COUNT} slices and hash-match them against the other sources."
echo "============================================================"
