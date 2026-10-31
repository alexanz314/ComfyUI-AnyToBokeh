#!/usr/bin/env bash
# download_weights.sh - fetch every weight the two published workflows need.
#
#   bash scripts/download_weights.sh
#
# Nothing is installed into your environment and nothing is overwritten: an item that is already
# on disk is skipped. Every path is a variable at the top and can be overridden:
#
#   COMFY_MODELS=/data/comfy/models bash scripts/download_weights.sh   # weights live elsewhere
#   PY=/opt/comfy/venv/bin/python   bash scripts/download_weights.sh   # another interpreter
#   SKIP_DRIVE=1 bash scripts/download_weights.sh                      # skip the Google Drive part
#   INSTALL_GDOWN=1 bash scripts/download_weights.sh                   # allow `pip install gdown`
#
# What it fetches (~10.5 GB total):
#   1. Any-to-Bokeh fine-tune (unet/ + vae/)   Google Drive, id from the upstream README (~7.4 GB)
#   2. stabilityai/stable-video-diffusion-img2vid-xt - ONLY the slice diffusers uses    (~1.2 GB)
#   3. Comfy-Org/Depth-Anything-3   -> models/geometry_estimation/                       (~1.3 GB)
#   4. Comfy-Org/sam3.1             -> models/checkpoints/          (sam-mask workflow only)
#
# All four sources are public (checked 2026-10-31: no gate on SVD-XT, DA3 or sam3.1), so no
# HF_TOKEN is needed. Export HF_TOKEN anyway if your network requires one.

set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_ROOT="${COMFY_ROOT:-$(cd "$PACK_DIR/../.." && pwd)}"   # <ComfyUI>/custom_nodes/<this pack>
COMFY_MODELS="${COMFY_MODELS:-$COMFY_ROOT/models}"
ANY2B_DIR="${ANY2B_DIR:-$COMFY_MODELS/any2bokeh}"
DRIVE_ID="${DRIVE_ID:-11UQcR7-GJtobPNKlF3f-q97xYX9pyEXb}"    # from the upstream README
SVD_REPO="${SVD_REPO:-stabilityai/stable-video-diffusion-img2vid-xt}"
DA3_REPO="${DA3_REPO:-Comfy-Org/Depth-Anything-3}"
DA3_FILE="${DA3_FILE:-geometry_estimation/depth_anything_3_mono_large.safetensors}"
SAM_REPO="${SAM_REPO:-Comfy-Org/sam3.1}"
SAM_FILE="${SAM_FILE:-checkpoints/sam3.1_multiplex_fp16.safetensors}"

SKIP_DRIVE="${SKIP_DRIVE:-0}"
SKIP_SVD="${SKIP_SVD:-0}"
SKIP_DA3="${SKIP_DA3:-0}"
SKIP_SAM="${SKIP_SAM:-0}"
INSTALL_GDOWN="${INSTALL_GDOWN:-0}"

PY="${PY:-}"
if [[ -z "$PY" ]]; then
  for cand in "$COMFY_ROOT/venv/bin/python" "$COMFY_ROOT/.venv/bin/python" "$(command -v python3 || true)"; do
    [[ -n "$cand" && -x "$cand" ]] && { PY="$cand"; break; }
  done
fi
[[ -n "${PY:-}" && -x "$PY" ]] || { echo "no python found (set PY=/path/to/python)"; exit 1; }

echo "== Any-to-Bokeh weights =="
echo "   pack:      $PACK_DIR"
echo "   ComfyUI:   $COMFY_ROOT"
echo "   models:    $COMFY_MODELS"
echo "   python:    $PY"
"$PY" -c 'import huggingface_hub, torch; print("   huggingface_hub", huggingface_hub.__version__, "| torch", torch.__version__)' \
  || { echo "!! this interpreter lacks huggingface_hub/torch: pip install -r $PACK_DIR/requirements.txt"; exit 1; }

# ------------------------------------------------------- 1. Any-to-Bokeh fine-tune (Google Drive) --
mkdir -p "$ANY2B_DIR"
if [[ "$SKIP_DRIVE" == "1" ]]; then
  echo "-- [1/4] SKIP_DRIVE=1: not fetching unet/ + vae/"
elif [[ -f "$ANY2B_DIR/vae/config.json" ]] && { [[ -f "$ANY2B_DIR/unet/config.json" ]] || [[ -f "$ANY2B_DIR/unet/unet/config.json" ]]; }; then
  echo "-- [1/4] unet/ and vae/ already present, skipping"
else
  echo "-- [1/4] Any-to-Bokeh checkpoints from Google Drive ($DRIVE_ID)"
  if ! "$PY" -c 'import gdown' 2>/dev/null && [[ "$INSTALL_GDOWN" == "1" ]]; then
    "$PY" -m pip install -q gdown
  fi
  if "$PY" -c 'import gdown' 2>/dev/null; then
    TMP_ZIP="$ANY2B_DIR/checkpoints_download"
    mkdir -p "$TMP_ZIP"
    # --fuzzy handles both the /file/d/ID and the open?id=ID URL forms
    "$PY" -m gdown --fuzzy "https://drive.google.com/file/d/$DRIVE_ID/view" -O "$TMP_ZIP/archive" || {
      echo "!! gdown failed (Drive throttling is common). Download the archive manually from the"
      echo "!! upstream README (https://github.com/vivoCameraResearch/any-to-bokeh) and unpack it so"
      echo "!! that $ANY2B_DIR/unet/ and $ANY2B_DIR/vae/ exist."
    }
    if [[ -f "$TMP_ZIP/archive" ]]; then
      echo "-- unpacking"
      if file "$TMP_ZIP/archive" | grep -qi zip; then
        (cd "$ANY2B_DIR" && unzip -o -q "$TMP_ZIP/archive")
      else
        tar -xf "$TMP_ZIP/archive" -C "$ANY2B_DIR" || echo "!! unknown archive format: $TMP_ZIP/archive"
      fi
      find "$ANY2B_DIR" -maxdepth 3 -type d | sort | sed 's/^/     /'
    fi
  else
    cat <<EOF
   gdown is not installed, and this script does not add packages to your environment unless you
   ask it to. Re-run with INSTALL_GDOWN=1, or download the archive yourself from the upstream
   README (https://github.com/vivoCameraResearch/any-to-bokeh) and unpack it so these exist:
       $ANY2B_DIR/unet/   (diffusers layout: config.json + diffusion_pytorch_model.safetensors)
       $ANY2B_DIR/vae/    (AutoencoderKLTemporalDecoder)
EOF
  fi
fi

# --------------------------------------------------------------------- 2-4. HuggingFace items ----
export ANY2B_DIR COMFY_MODELS SVD_REPO DA3_REPO DA3_FILE SAM_REPO SAM_FILE
export SKIP_SVD SKIP_DA3 SKIP_SAM HF_TOKEN="${HF_TOKEN:-}"

set +e
"$PY" - <<'PYEOF'
import os
import shutil
import sys

from huggingface_hub import hf_hub_download, snapshot_download

models = os.environ["COMFY_MODELS"]
any2b = os.environ["ANY2B_DIR"]
svd_dir = os.path.join(any2b, "svd_xt")
token = os.environ.get("HF_TOKEN") or None
problems = []


def size_gb(path):
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in (".cache", "huggingface")]
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 2 ** 30


def report(ok, label, where, note=""):
    print(("   OK   " if ok else "   MISS ") + f"{label}  ->  {where}" + (f"   ({note})" if note else ""))
    if not ok:
        problems.append(label)


def fetch_file(repo, filename, subdir, label):
    """hf_hub_download into a temp dir, then move the single file into models/<subdir>/."""
    dest_dir = os.path.join(models, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(filename))
    if os.path.isfile(dest):
        print(f"-- {label}: {os.path.basename(dest)} already present, skipping")
    else:
        print(f"-- {label}: {repo} -> {dest_dir}")
        tmp = os.path.join(models, f"_a2b_tmp_{subdir.replace('/', '_')}")
        try:
            hf_hub_download(repo_id=repo, filename=filename, local_dir=tmp, token=token)
            shutil.move(os.path.join(tmp, filename), dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    report(os.path.isfile(dest), label, dest, f"{os.path.getsize(dest) / 2 ** 30:.2f} GiB"
           if os.path.isfile(dest) else "")


# --- 2. SVD-XT: only the slice diffusers actually loads (~1.2 GiB of a 21.5 GiB repo) -------------
# The unet/ and vae/ weights are replaced by the Any-to-Bokeh fine-tune (they are passed to
# from_pretrained as objects), so only their config.json is kept as a safety net; the single-file
# svd_xt*.safetensors checkpoints are for ComfyUI's native loader, not for diffusers; and upstream
# loads with variant="fp16", so only image_encoder/model.fp16.safetensors is needed.
if os.environ["SKIP_SVD"] == "1":
    print("-- [2/4] SKIP_SVD=1")
else:
    if os.path.isfile(os.path.join(svd_dir, "model_index.json")):
        print(f"-- [2/4] SVD-XT slice already present ({size_gb(svd_dir):.2f} GiB), skipping")
    else:
        print(f"-- [2/4] {os.environ['SVD_REPO']} -> {svd_dir} (slice only, ~1.2 GiB)")
        snapshot_download(
            repo_id=os.environ["SVD_REPO"], local_dir=svd_dir, token=token,
            allow_patterns=[
                "model_index.json", "LICENSE.md", "scheduler/*", "feature_extractor/*",
                "image_encoder/config.json", "image_encoder/model.fp16.safetensors",
                "unet/config.json", "vae/config.json",
            ],
        )
    missing = [f for f in ("model_index.json", "scheduler/scheduler_config.json",
                           "feature_extractor/preprocessor_config.json", "image_encoder/config.json",
                           "image_encoder/model.fp16.safetensors")
               if not os.path.isfile(os.path.join(svd_dir, f))]
    report(not missing, "[2/4] SVD-XT slice", svd_dir,
           f"{size_gb(svd_dir):.2f} GiB" if not missing else f"missing: {missing}")

# --- 3. Depth Anything 3 (core LoadDA3Model reads models/geometry_estimation/) -------------------
if os.environ["SKIP_DA3"] == "1":
    print("-- [3/4] SKIP_DA3=1")
else:
    fetch_file(os.environ["DA3_REPO"], os.environ["DA3_FILE"], "geometry_estimation",
               "[3/4] Depth Anything 3 (mono large)")

# --- 4. SAM3.1 (only the sam-mask workflow needs it; core CheckpointLoaderSimple) ----------------
if os.environ["SKIP_SAM"] == "1":
    print("-- [4/4] SKIP_SAM=1")
else:
    fetch_file(os.environ["SAM_REPO"], os.environ["SAM_FILE"], "checkpoints",
               "[4/4] SAM3.1 multiplex fp16")

if problems:
    print("\n!! incomplete:", ", ".join(problems))
sys.exit(1 if problems else 0)
PYEOF
HF_RC=$?
set -e

cat <<EOF

== layout expected by the "Any-to-Bokeh: Load Model" node ==
  $ANY2B_DIR/unet/     their fine-tuned SVD UNet (config.json + diffusion_pytorch_model.safetensors)
  $ANY2B_DIR/vae/      AutoencoderKLTemporalDecoder
  $ANY2B_DIR/svd_xt/   $SVD_REPO (must contain model_index.json)
  $COMFY_MODELS/geometry_estimation/$(basename "$DA3_FILE")
  $COMFY_MODELS/checkpoints/$(basename "$SAM_FILE")   <- sam-mask workflow only

Relative paths in the Load Model node resolve against $COMFY_MODELS, so
"any2bokeh/unet", "any2bokeh/vae" and "any2bokeh/svd_xt" work as-is.
Weights somewhere else: export ANY2BOKEH_MODELS=/path/to/dir (colon-separated list), or type
absolute paths into the node.
EOF

exit $HF_RC
