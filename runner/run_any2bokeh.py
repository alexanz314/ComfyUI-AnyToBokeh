#!/usr/bin/env python
"""Any-to-Bokeh inference runner.

Runs in an ISOLATED venv (upstream pins diffusers==0.25.0 / transformers==4.32.1, which are far
older than what ComfyUI ships), so the ComfyUI node talks to this script as a subprocess.

All tensor preparation (disparity normalisation, CoC, MPI mask, frame grouping, overlap weights)
happens on the ComfyUI side in a2b/prepare.py. This script only loads the model and reproduces
the loop from upstream test/inference_demo.py.

Usage:
    python run_any2bokeh.py --job /path/to/job.json

job.json keys:
    tensors_npz, unet_path, vae_path, svd_path, out_npy,
    height, width, group_frames, overlap_frames, infer_clip_num,
    num_inference_steps, motion_bucket_id, fps, noise_aug_strength, seed,
    dtype ("float16"|"float32"), svd_variant ("fp16"|""), device

Prints exactly one JSON status line on stdout as the last line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "any2bokeh_repo"))


def _fail(msg: str, **extra) -> int:
    print(json.dumps({"ok": False, "error": msg, **extra}))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="path to job JSON")
    args = ap.parse_args()

    try:
        with open(args.job, "r", encoding="utf-8") as fh:
            job = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"cannot read job file: {exc}")

    for key in ("tensors_npz", "unet_path", "vae_path", "svd_path", "out_npy"):
        if not job.get(key):
            return _fail(f"job is missing required key {key!r}")

    # Must be set before CUDA initialises. The subprocess shares the GPU with ComfyUI, which keeps
    # its own context and cached blocks; expandable segments markedly reduces fragmentation-driven
    # OOM during the VAE decode of 576x1024 frames.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        import numpy as np
        import torch

        if REPO not in sys.path:
            sys.path.insert(0, REPO)

        # Upstream pins diffusers==0.25.0; a2b_compat maps the module paths that moved in 0.39+
        # so the vendored code runs unmodified in whatever interpreter ComfyUI uses.
        import a2b_compat
        for note in a2b_compat.install():
            if os.environ.get("A2B_DEBUG"):
                print(f"[a2b-compat] {note}", file=sys.stderr)

        from a2b_models.unet import UNetSpatioTemporalConditionModel
        from a2b_models.vae import AutoencoderKLTemporalDecoder
        from a2b_pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline
    except Exception as exc:  # noqa: BLE001
        return _fail(f"import error: {exc}", traceback=traceback.format_exc())

    dtype = torch.float16 if job.get("dtype", "float16") == "float16" else torch.float32
    device = job.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    height = int(job.get("height", 576))
    width = int(job.get("width", 1024))
    group_frames = int(job.get("group_frames", 8))
    overlap_frames = int(job.get("overlap_frames", 4))
    infer_clip_num = int(job.get("infer_clip_num", 2))
    variant = job.get("svd_variant", "fp16") or None

    try:
        data = np.load(job["tensors_npz"])
        pixel_values = torch.from_numpy(data["pixel_values"])
        coc = torch.from_numpy(data["coc"])
        mpi_mask = torch.from_numpy(data["mpi_mask"])
        k = torch.from_numpy(data["k"])
        weight = torch.from_numpy(data["weight"])
    except Exception as exc:  # noqa: BLE001
        return _fail(f"cannot load tensors: {exc}", traceback=traceback.format_exc())

    try:
        # Upstream calls from_pretrained(checkpoints/unet, subfolder="unet"), i.e. it expects
        # <path>/unet/config.json. The released archive actually puts config.json directly in
        # <path>/, so detect the layout instead of hard-coding it.
        unet_path = job["unet_path"]
        unet_kwargs = {}
        if os.path.exists(os.path.join(unet_path, "config.json")):
            pass
        elif os.path.exists(os.path.join(unet_path, "unet", "config.json")):
            unet_kwargs["subfolder"] = "unet"
        else:
            return _fail(f"no config.json in {unet_path} (looked in '.' and './unet')")

        svd_path = job["svd_path"]
        if not os.path.exists(os.path.join(svd_path, "model_index.json")):
            return _fail(
                f"{svd_path} has no model_index.json - this must be the diffusers layout of "
                f"stabilityai/stable-video-diffusion-img2vid-xt (gated repo, needs HF_TOKEN)"
            )

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            unet_path, torch_dtype=dtype, low_cpu_mem_usage=False, local_files_only=True, **unet_kwargs,
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            job["vae_path"], local_files_only=True, torch_dtype=dtype,
        )
        pipe_kwargs = dict(unet=unet, vae=vae, low_cpu_mem_usage=False, torch_dtype=dtype, local_files_only=True)
        if variant:
            pipe_kwargs["variant"] = variant
        pipe = StableVideoDiffusionPipeline.from_pretrained(svd_path, **pipe_kwargs)
        # The released SVD VAE config sets force_upcast=True. Two consequences on modern diffusers:
        # the weights stay in fp32, and ModelMixin.to(dtype) silently IGNORES the cast for such a
        # model - so the pipeline feeds fp16 tensors into fp32 convs and dies in the encoder with
        #   RuntimeError: Input type (c10::Half) and bias type (float) should be the same
        # Clear the flag and cast the parameters directly, instead of pinning diffusers 0.25.
        if getattr(pipe.vae.config, "force_upcast", False):
            try:
                pipe.vae.register_to_config(force_upcast=False)
            except Exception:  # noqa: BLE001
                pass
            for mod in pipe.vae.modules():
                for p in mod.parameters(recurse=False):
                    p.data = p.data.to(dtype)
                for b in mod.buffers(recurse=False):
                    b.data = b.data.to(dtype)
            print(f"[a2b] vae force_upcast cleared, parameters cast to {dtype}", file=sys.stderr)
        vae_dtype = next(pipe.vae.parameters()).dtype
        if vae_dtype != dtype:
            print(f"[a2b] casting vae {vae_dtype} -> {dtype}", file=sys.stderr)
            pipe.vae.to(dtype)
        # Keep idle submodels in RAM instead of pinning everything to the GPU. While the VAE encodes
        # the reference and CoC conditions, the ~3.8 GB UNet and the ~1.2 GB CLIP image encoder are
        # not used at all - on a 24 GB card that dead weight is precisely what pushes the encoder's
        # activations over the limit (observed: 17.22 GiB allocated, 2.25 GiB request, OOM).
        # Set A2B_NO_OFFLOAD=1 to disable, e.g. on a card with plenty of VRAM.
        if device.startswith("cuda") and os.environ.get("A2B_NO_OFFLOAD") != "1":
            try:
                pipe.enable_model_cpu_offload()
                print("[a2b] model cpu offload enabled (idle submodels stay in RAM)", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"[a2b] cpu offload unavailable ({exc}); falling back to pipe.to(device)", file=sys.stderr)
                pipe.to(device)
        else:
            pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"model load failed: {exc}", traceback=traceback.format_exc())

    # --- loop reproduced from test/inference_demo.py -----------------------------------------
    frames_out = []
    try:
        latent_cache = None
        n_groups = coc.shape[0]
        for i in range(0, n_groups, infer_clip_num):
            is_last_batch = i + infer_clip_num >= n_groups
            w = weight[i:i + infer_clip_num].to(dtype).to(device, non_blocking=True)
            control = coc[i:i + infer_clip_num].to(device, non_blocking=True)
            ref = pixel_values[i:i + infer_clip_num].to(device, non_blocking=True)
            in_k = k[i:i + infer_clip_num].to(dtype).to(device, non_blocking=True)
            in_mpi = mpi_mask[i:i + infer_clip_num].to(dtype).to(device, non_blocking=True)

            video_frames, latent_cache = pipe(
                control,
                ref,
                input_k=in_k,
                mpi_mask=in_mpi,
                latent_cache=latent_cache,
                is_last_batch=is_last_batch,
                overlap_frame=overlap_frames,
                weight=w,
                height=height,
                width=width,
                num_frames=group_frames,
                # Upstream decodes the whole group at once, which OOMs a 24 GB card at 576x1024.
                # Decoding fewer frames at a time costs speed, not quality.
                decode_chunk_size=int(job.get("decode_chunk_size", 2)),
                motion_bucket_id=int(job.get("motion_bucket_id", 127)),
                fps=int(job.get("fps", 7)),
                noise_aug_strength=float(job.get("noise_aug_strength", 0.0)),
                num_inference_steps=int(job.get("num_inference_steps", 1)),
                generator=torch.Generator(device="cpu").manual_seed(int(job.get("seed", 0))),
                batch_size=infer_clip_num if w.shape[0] == infer_clip_num else 1,
            )
            frames_out.extend(video_frames[0])
    except Exception as exc:  # noqa: BLE001
        return _fail(f"inference failed: {exc}", traceback=traceback.format_exc())

    try:
        arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.float32) / 255.0 for f in frames_out])
        os.makedirs(os.path.dirname(os.path.abspath(job["out_npy"])), exist_ok=True)
        np.save(job["out_npy"], arr.astype(np.float16))
    except Exception as exc:  # noqa: BLE001
        return _fail(f"cannot save output: {exc}", traceback=traceback.format_exc())

    print(json.dumps({"ok": True, "out": job["out_npy"], "frames": int(arr.shape[0])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
