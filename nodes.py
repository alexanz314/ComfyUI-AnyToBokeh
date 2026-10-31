"""ComfyUI nodes for Any-to-Bokeh (vivoCameraResearch/any-to-bokeh, ICLR 2026).

Design notes
------------
* Depth and masks are NOT reimplemented here. Use the ComfyUI core nodes:
    - LoadDA3Model -> DA3Inference  (comfy_extras/nodes_depth_anything_3.py) -> DA3_GEOMETRY
    - CheckpointLoaderSimple(sam3.1) -> CLIPTextEncode -> SAM3_Detect          -> MASK
  This pack only converts those into the exact tensors Any-to-Bokeh expects and runs its model.

* Upstream weights are diffusers-format folders for a MODIFIED SVD UNet (extra CoC / MPI-mask / k
  conditioning). They cannot be expressed as a single ComfyUI .safetensors, so they are loaded by
  upstream code through runner/run_any2bokeh.py.

* Upstream pins diffusers==0.25.0 and transformers==4.32.1. ComfyUI ships much newer versions, so
  any2bokeh_repo/a2b_compat.py aliases the module paths that moved and the model runs in a
  SUBPROCESS (default backend) to keep ~7.5 GB of weights out of ComfyUI's own allocator.
  Downgrading diffusers inside the ComfyUI venv is not recommended: it can break other custom nodes.

* This is the v2 pack: the subset of nodes the two workflows in ./workflows use. Node class names
  are unchanged from v1, so ComfyUI-AnyToBokeh and ComfyUI-AnyToBokeh_v2 must NOT both be enabled
  (ComfyUI keeps whichever pack it loads first and warns about the duplicates).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

import folder_paths
import comfy.model_management as mm

from .a2b import prepare as P

A2B_DISPARITY = "A2B_DISPARITY"
A2B_FOCUS = "A2B_FOCUS"
A2B_PIPE = "A2B_PIPE"

MODELS_SUBDIR = "any2bokeh"
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner", "run_any2bokeh.py")


def _search_roots() -> list[str]:
    """Directories scanned for diffusers-format checkpoint folders.

    ComfyUI's own models dir first, then ANY2BOKEH_MODELS (colon-separated), so weights can live
    on any drive. Options are shown as absolute paths.
    """
    roots = [os.path.join(folder_paths.models_dir, MODELS_SUBDIR)]
    env = os.environ.get("ANY2BOKEH_MODELS", "")
    roots += [p for p in env.split(":") if p]
    out = []
    for r in roots:
        if r and os.path.isdir(r) and r not in out:
            out.append(r)
    return out


def _default_paths() -> dict[str, str]:
    """Where the weights are expected: models/any2bokeh/{unet,vae,svd_xt}.

    These are diffusers-format DIRECTORIES (config.json + diffusion_pytorch_model.safetensors), so
    they cannot live as loose files in models/unet or models/vae - ComfyUI's own loaders list those
    folders and a bare 7.5 GB safetensors there would show up in UNETLoader and fail to load.
    """
    root = os.path.join(folder_paths.models_dir, MODELS_SUBDIR)
    return {
        "unet": os.path.join(root, "unet"),
        "vae": os.path.join(root, "vae"),
        "svd": os.path.join(root, "svd_xt"),
    }


def _resolve(path: str, what: str, need_model_index: bool = False) -> str:
    path = os.path.expanduser((path or "").strip())
    if not path or path.startswith("<"):
        raise ValueError(f"{what}: path is not set")
    if not os.path.isabs(path):
        path = os.path.join(folder_paths.models_dir, path)
    if not os.path.isdir(path):
        roots = _search_roots()
        raise ValueError(f"{what}: no directory at {path} (also searched: {roots})")
    if need_model_index:
        if not os.path.exists(os.path.join(path, "model_index.json")):
            raise ValueError(
                f"{what}: {path} has no model_index.json. Expected the diffusers layout of "
                "stabilityai/stable-video-diffusion-img2vid-xt (public repo, no token needed). "
                "Run: bash scripts/download_weights.sh -- it fetches only the ~1.2 GiB slice that "
                "is actually used, not the full 21.5 GiB."
            )
    elif not os.path.exists(os.path.join(path, "config.json")) and \
            not os.path.exists(os.path.join(path, "unet", "config.json")):
        raise ValueError(f"{what}: {path} has no config.json (neither in '.' nor in './unet')")
    return path


class AnyToBokehDisparity:
    """Depth (DA3 geometry or a depth IMAGE) -> clip-normalised uint8 disparity, as upstream expects."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "convention": (
                    ["depth: larger = farther (DA3 / MoGe)", "disparity: larger = closer (VDA / DAv2)"],
                    {"tooltip": "DA3 geometry 'depth' follows the MoGe convention (larger = farther) and must be "
                                "inverted. Video-Depth-Anything and Depth Anything V2 relative output is already "
                                "disparity-like (larger = closer) - that is what upstream consumes directly."},
                ),
            },
            "optional": {
                "da3_geometry": ("DA3_GEOMETRY", {"tooltip": "Preferred: raw DA3Inference output. Uses geometry['depth']."}),
                "depth_image": ("IMAGE", {"tooltip": "Fallback: a greyscale depth/disparity IMAGE (e.g. DA3Render). "
                                                     "Lossy - the raw geometry input is better."}),
            },
        }

    RETURN_TYPES = (A2B_DISPARITY, "IMAGE")
    RETURN_NAMES = ("disparity", "preview")
    FUNCTION = "run"
    CATEGORY = "AnyToBokeh"

    def run(self, convention, da3_geometry=None, depth_image=None):
        if da3_geometry is None and depth_image is None:
            raise ValueError("connect either da3_geometry (DA3Inference) or depth_image")
        larger_is_farther = convention.startswith("depth:")

        if da3_geometry is not None:
            if "depth" not in da3_geometry:
                raise ValueError("DA3_GEOMETRY has no 'depth' key")
            depth = da3_geometry["depth"]
            if isinstance(depth, torch.Tensor):
                depth = depth.detach().to(torch.float32).cpu()
            else:
                depth = torch.as_tensor(np.asarray(depth), dtype=torch.float32)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 4:  # (B,C,H,W)
                depth = depth[:, 0]
        else:
            img = depth_image.detach().to(torch.float32).cpu()
            depth = img[..., 0] if img.ndim == 4 else img
            if img.max() <= 1.5:
                depth = depth * 255.0

        if depth.ndim != 3:
            raise ValueError(f"expected (N,H,W) depth, got {tuple(depth.shape)}")

        disp_u8 = P.depth_to_disparity(depth, larger_is_farther=larger_is_farther)
        preview = (disp_u8.float() / 255.0).unsqueeze(-1).repeat(1, 1, 1, 3)
        return ({"disp_u8": disp_u8, "n": int(disp_u8.shape[0])}, preview)


class AnyToBokehFocusFromMask:
    """Focal plane per frame = mean disparity inside the subject mask (upstream `disp_concept`)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disparity": (A2B_DISPARITY,),
                "mask": ("MASK", {"tooltip": "Mask of the object that should be in focus. SAM3_Detect gives a "
                                             "per-frame mask from a text prompt; SAM3_VideoTrack -> SAM3_TrackToMask "
                                             "gives a temporally tracked one."}),
            },
            "optional": {
                "empty_mask_fallback": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                                  "tooltip": "Unused in the normal path: frames whose mask is empty "
                                                             "carry the focal plane from the nearest detection. "
                                                             "If the mask is empty on EVERY frame the node raises "
                                                             "instead of silently blurring everything."}),
            },
        }

    RETURN_TYPES = (A2B_FOCUS, "IMAGE", "STRING")
    RETURN_NAMES = ("focus", "mask_preview", "report")
    FUNCTION = "run"
    CATEGORY = "AnyToBokeh"

    def run(self, disparity, mask, empty_mask_fallback=0.5):
        disp_u8 = disparity["disp_u8"]
        m = mask.detach().to(torch.float32).cpu()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        if m.shape[0] != disp_u8.shape[0]:
            raise ValueError(f"mask has {m.shape[0]} frames, disparity has {disp_u8.shape[0]}")
        focus, empty = P.focus_from_mask(disp_u8, m, fallback=float(empty_mask_fallback))
        coverage = float((m > 0.5).float().mean())
        report = (f"frames={focus.shape[0]} focus min/mean/max = "
                  f"{float(focus.min()):.3f}/{float(focus.mean()):.3f}/{float(focus.max()):.3f} "
                  f"| mask coverage={coverage:.3f}"
                  + (f" | EMPTY MASK on {len(empty)} frames -> carried from nearest detection" if empty else ""))
        preview = (m > 0.5).float().unsqueeze(-1).repeat(1, 1, 1, 3)
        return ({"focus": focus, "n": int(focus.shape[0])}, preview, report)


class AnyToBokehLoadModel:
    """Resolve the diffusers-format checkpoint folders (they are directories, not single files)."""

    @classmethod
    def INPUT_TYPES(cls):
        d = _default_paths()
        return {
            "required": {
                "unet_dir": ("STRING", {"default": d["unet"], "multiline": False,
                                        "tooltip": "Directory with config.json + diffusion_pytorch_model.safetensors "
                                                   "of their fine-tuned SVD UNet. Relative paths resolve against "
                                                   "<ComfyUI>/models."}),
                "vae_dir": ("STRING", {"default": d["vae"], "multiline": False,
                                       "tooltip": "Directory for AutoencoderKLTemporalDecoder (config.json + safetensors)."}),
                "svd_dir": ("STRING", {"default": d["svd"], "multiline": False,
                                       "tooltip": "stabilityai/stable-video-diffusion-img2vid-xt in diffusers layout "
                                                  "(must contain model_index.json). Gated repo - accept the licence "
                                                  "and download with HF_TOKEN."}),
                "dtype": (["float16", "float32"], {"default": "float16"}),
                "svd_variant": (["fp16", "none"], {"default": "fp16",
                                                   "tooltip": "Passed as variant= to from_pretrained. Use 'none' if the "
                                                              "SVD folder has no fp16 variant files."}),
            },
        }

    RETURN_TYPES = (A2B_PIPE,)
    RETURN_NAMES = ("a2b_model",)
    FUNCTION = "run"
    CATEGORY = "AnyToBokeh"

    def run(self, unet_dir, vae_dir, svd_dir, dtype, svd_variant):
        cfg = {
            "unet_path": _resolve(unet_dir, "unet_dir"),
            "vae_path": _resolve(vae_dir, "vae_dir"),
            "svd_path": _resolve(svd_dir, "svd_dir", need_model_index=True),
            "dtype": dtype,
            "svd_variant": "" if svd_variant == "none" else svd_variant,
        }
        return (cfg,)


class AnyToBokehPreviewCoC:
    """Debug: visualise the CoC map and the MPI region mask actually fed to the model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disparity": (A2B_DISPARITY,),
                "focus": (A2B_FOCUS,),
                "k": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 64.0, "step": 0.5}),
                "mpi_alpha": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 3.0, "step": 0.1}),
            },
            "optional": {"k_per_frame": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("coc", "mpi_regions", "report")
    FUNCTION = "run"
    CATEGORY = "AnyToBokeh"

    def run(self, disparity, focus, k, mpi_alpha, k_per_frame=""):
        disp_u8 = disparity["disp_u8"]
        f = focus["focus"]
        n = disp_u8.shape[0]
        if f.shape[0] != n:
            raise ValueError(f"focus has {f.shape[0]} values but disparity has {n} frames")
        d = disp_u8.float()
        cocs, mpis = [], []
        for i in range(n):
            fi = float(f[i].item())
            coc = torch.abs(d[i] - fi * 255.0) / 255.0
            cocs.append(coc)
            mpis.append(P.gen_mpi_mask(d[i] / 255.0, fi, alpha=float(mpi_alpha)).float() / (P.MPI_SEGMENTS - 1))
        coc_img = torch.stack(cocs).unsqueeze(-1).repeat(1, 1, 1, 3)
        mpi_img = torch.stack(mpis).unsqueeze(-1).repeat(1, 1, 1, 3)
        kk = P.parse_k_list(k_per_frame, n, float(k))
        report = f"frames={n} k min/max={kk.min():.2f}/{kk.max():.2f} coc mean={coc_img.mean():.3f}"
        return (coc_img, mpi_img, report)


class AnyToBokehGenerate:
    """Run the model: sharp clip + disparity + focal plane -> refocused (bokeh) clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a2b_model": (A2B_PIPE,),
                "images": ("IMAGE", {"tooltip": "The sharp source clip (all-in-focus master)."}),
                "disparity": (A2B_DISPARITY,),
                "focus": (A2B_FOCUS,),
                "k": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 64.0, "step": 0.5,
                                "tooltip": "Blur strength. Upstream demo CSV uses 16."}),
                "height": ("INT", {"default": P.DEFAULT_HEIGHT, "min": 64, "max": 2048, "step": 8}),
                "width": ("INT", {"default": P.DEFAULT_WIDTH, "min": 64, "max": 2048, "step": 8}),
                "backend": (["subprocess: isolated venv (recommended)", "in-process: this venv"],
                            {"default": "subprocess: isolated venv (recommended)"}),
            },
            "optional": {
                "k_per_frame": ("STRING", {"default": "", "tooltip": "Optional per-frame blur strength, e.g. '16,16,24' "
                                                                     "or one value per frame. Empty = constant k."}),
                "group_frames": ("INT", {"default": P.DEFAULT_GROUP_FRAMES, "min": 2, "max": 32, "step": 1}),
                "overlap_frames": ("INT", {"default": P.DEFAULT_OVERLAP_FRAMES, "min": 0, "max": 16, "step": 1}),
                "infer_clip_num": ("INT", {"default": P.DEFAULT_INFER_CLIP_NUM, "min": 1, "max": 2, "step": 1,
                                           "tooltip": "Upstream supports only 2."}),
                "num_inference_steps": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                                                "tooltip": "Upstream uses 1 (single deterministic forward)."}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 255, "step": 1}),
                "fps": ("INT", {"default": 7, "min": 1, "max": 60, "step": 1}),
                "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "mpi_alpha": ("FLOAT", {"default": P.MPI_ALPHA_INFER, "min": 0.5, "max": 3.0, "step": 0.1}),
                "python_bin": ("STRING", {"default": "", "multiline": False,
                                         "tooltip": "Interpreter for the runner subprocess. Empty = auto-detect "
                                                    "(a venv inside this pack if present, else the ComfyUI one)."}),
                "timeout_minutes": ("INT", {"default": 120, "min": 1, "max": 1440, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "coc_preview", "report")
    FUNCTION = "run"
    CATEGORY = "AnyToBokeh"

    def _auto_python(self) -> str:
        """Interpreter for the runner subprocess.

        Prefers an isolated venv inside the pack if one exists, otherwise falls back to the
        interpreter ComfyUI runs on: a2b_compat maps the diffusers module paths that moved since the
        upstream 0.25 pin, so the current venv works. Verified on diffusers 0.39.0: unet, vae and
        the pipeline all load, and the unet checkpoint matches the architecture exactly
        (1830 tensors, 0 missing / 0 unexpected / 0 shape mismatches). Keeping the run in a
        subprocess still holds ~7.5 GB of weights out of ComfyUI's own allocator.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, "venv", "bin", "python"),
                     os.path.join(here, ".venv", "bin", "python")):
            if os.path.exists(cand):
                return cand
        return sys.executable

    def run(self, a2b_model, images, disparity, focus, k, height, width, backend,
            k_per_frame="", group_frames=P.DEFAULT_GROUP_FRAMES, overlap_frames=P.DEFAULT_OVERLAP_FRAMES,
            infer_clip_num=P.DEFAULT_INFER_CLIP_NUM, num_inference_steps=1, motion_bucket_id=127, fps=7,
            noise_aug_strength=0.0, seed=0, mpi_alpha=P.MPI_ALPHA_INFER, python_bin="", timeout_minutes=120):
        img = images.detach().to(torch.float32).cpu()
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"images must be (N,H,W,3), got {tuple(img.shape)}")
        n = int(img.shape[0])
        disp_u8 = disparity["disp_u8"]
        f = focus["focus"]
        if disp_u8.shape[0] != n:
            raise ValueError(f"disparity has {disp_u8.shape[0]} frames, images have {n}")
        if f.shape[0] != n:
            raise ValueError(f"focus has {f.shape[0]} values, images have {n}")
        if overlap_frames >= group_frames:
            raise ValueError("overlap_frames must be smaller than group_frames")

        kk = P.parse_k_list(k_per_frame, n, float(k))
        groups = P.build_groups(
            img, disp_u8, f, kk,
            height=int(height), width=int(width),
            group_frames=int(group_frames), overlap_frames=int(overlap_frames),
            mpi_alpha=float(mpi_alpha),
        )

        # Upstream's decode_latents only appends the tail frames when a latent_cache already
        # exists, i.e. the FIRST batch must not also be the LAST one. With infer_clip_num=2 that
        # means at least 3 groups; fewer groups die deep inside torch.cat / reshape with
        # "Expected size 1 but got size 0" or "shape [-1, N, ...] is invalid", which says nothing
        # about the real cause. A single group fails even earlier: latents[1:2] is read
        # unconditionally when there is no cache.
        min_groups = int(infer_clip_num) + 1
        step = int(group_frames) - int(overlap_frames)
        # num_groups(n) == ceil((n - overlap) / step), so this is the smallest n reaching min_groups
        min_frames = int(overlap_frames) + (min_groups - 1) * step + 1
        if int(infer_clip_num) != 2 or groups["num_groups"] < min_groups:
            raise ValueError(
                f"clip too short for upstream's overlap blending: {n} frames with "
                f"group_frames={group_frames}/overlap_frames={overlap_frames} give "
                f"{groups['num_groups']} group(s), but the first batch of {infer_clip_num} must not "
                f"also be the last one, so at least {min_groups} groups are needed "
                f"(>= {min_frames} frames at these settings). Use a longer clip, or lower "
                f"group_frames. infer_clip_num must stay 2 - upstream supports nothing else."
            )

        # CoC preview straight from the tensors that go into the model: (G,F,H,W) -> (G*F,H,W,3)
        coc01 = ((groups["coc"][:, :, 0] + 1.0) * 0.5).clamp(0, 1)
        coc_preview = coc01.unsqueeze(-1).repeat(1, 1, 1, 1, 3).reshape(-1, height, width, 3)

        job = {
            "unet_path": a2b_model["unet_path"], "vae_path": a2b_model["vae_path"],
            "svd_path": a2b_model["svd_path"], "dtype": a2b_model["dtype"],
            "svd_variant": a2b_model.get("svd_variant", "fp16"),
            "height": int(height), "width": int(width),
            "group_frames": int(group_frames), "overlap_frames": int(overlap_frames),
            "infer_clip_num": int(infer_clip_num), "num_inference_steps": int(num_inference_steps),
            "motion_bucket_id": int(motion_bucket_id), "fps": int(fps),
            "noise_aug_strength": float(noise_aug_strength), "seed": int(seed),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        }

        tmp = tempfile.mkdtemp(prefix="a2b_", dir=folder_paths.get_temp_directory())
        npz_path = os.path.join(tmp, "groups.npz")
        out_path = os.path.join(tmp, "out.npy")
        np.savez_compressed(
            npz_path,
            pixel_values=groups["pixel_values"].to(torch.float16).numpy(),
            coc=groups["coc"].to(torch.float16).numpy(),
            mpi_mask=groups["mpi_mask"].to(torch.float16).numpy(),
            k=groups["k"].numpy(), weight=groups["weight"].numpy(),
        )
        job["tensors_npz"], job["out_npy"] = npz_path, out_path
        job_path = os.path.join(tmp, "job.json")
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump(job, fh)

        if backend.startswith("subprocess"):
            py = python_bin.strip() or self._auto_python()
            if not os.path.exists(py):
                raise ValueError(f"python_bin not found: {py}")
            mm.unload_all_models()
            mm.soft_empty_cache()
            # stdout carries the single JSON status line and is captured; stderr (the tqdm progress
            # bar, progress lines, warnings) is deliberately NOT captured, so it appears in the
            # ComfyUI console while the render runs.
            proc = subprocess.run(
                [py, RUNNER, "--job", job_path],
                stdout=subprocess.PIPE, text=True, timeout=int(timeout_minutes) * 60,
            )
            tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
            status = None
            if tail:
                try:
                    status = json.loads(tail[-1])
                except json.JSONDecodeError:
                    status = None
            if status is None or not status.get("ok"):
                err = (status or {}).get("error", "no status line from runner")
                tb = (status or {}).get("traceback", "")
                raise RuntimeError(f"Any-to-Bokeh runner failed: {err}\n{tb}")
        else:
            import contextlib
            import io
            import importlib.util
            spec = importlib.util.spec_from_file_location("a2b_runner", RUNNER)
            mod = importlib.util.module_from_spec(spec)
            argv_backup = sys.argv
            buf = io.StringIO()
            sys.argv = [RUNNER, "--job", job_path]
            try:
                # capture the runner's own stdout so its JSON status line (with the real error and
                # traceback) reaches the user instead of a generic "see console output above"
                with contextlib.redirect_stdout(buf):
                    spec.loader.exec_module(mod)
                    code = mod.main()
            finally:
                sys.argv = argv_backup
            text = buf.getvalue()
            sys.stdout.write(text)
            status = None
            tail = [ln for ln in text.strip().splitlines() if ln.strip()]
            if tail:
                try:
                    status = json.loads(tail[-1])
                except json.JSONDecodeError:
                    status = None
            if code != 0 or (status is not None and not status.get("ok")):
                err = (status or {}).get("error", "runner exited non-zero without a status line")
                tb = (status or {}).get("traceback", "")
                raise RuntimeError(f"Any-to-Bokeh runner failed: {err}\n{tb}\n--- runner stdout ---\n{text[-3000:]}")

        arr = np.load(out_path).astype(np.float32)
        out = torch.from_numpy(arr)
        report = (f"frames_in={n} frames_out={out.shape[0]} groups={groups['num_groups']} "
                  f"k={float(kk.min()):.1f}..{float(kk.max()):.1f} size={width}x{height}")
        return (out, coc_preview, report)


class AnyToBokehFocusControl:
    """Author the focal-plane scenario explicitly: keyframes as points, values or near/mid/far.

    Replaces guesswork with a script you write: one 'frame:target' per line. A single entry means a
    FIXED focal plane for the whole clip; several entries give a scenario with the chosen
    interpolation. The curve output renders the scenario so it can be checked before rendering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disparity": (A2B_DISPARITY,),
                "keyframes": ("STRING", {
                    "default": "0:0.5",
                    "multiline": False,
                    "tooltip": "Advanced/plain-text form: 'frame:target' pairs separated by ';' or newlines, "
                               "e.g. '0:234;120:139;end:31'. frame = int index or 'end'; target = plane value "
                               "(0..1 or 0..255), 'near/mid/far', or a point 'x,y' in 0..1 of the frame. "
                               "A single entry = fixed focus for the whole clip. Editable on the node's curve.",
                }),
                "interpolation": (["smooth", "linear", "hold"], {"default": "smooth",
                                                                  "tooltip": "smooth = cosine ease between keys, "
                                                                             "linear = straight ramps, "
                                                                             "hold = step changes at keys"}),
            },
            "optional": {
                "clip_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                        "tooltip": "Clip length for the node's curve x-axis (0 = auto). "
                                                   "Drawing only: the server always uses the real disparity length."}),
            },
        }

    RETURN_TYPES = (A2B_FOCUS, "IMAGE", "STRING")
    RETURN_NAMES = ("focus", "curve", "report")
    FUNCTION = "run"
    CATEGORY = "AnyToBokeh"

    def run(self, disparity, keyframes, interpolation, clip_frames=0):
        disp_u8 = disparity["disp_u8"]
        f, entries = P.parse_focus_keyframes(keyframes, disp_u8, interp=interpolation)
        curve = P.focus_curve_image(f, entries).unsqueeze(0)
        detail = ", ".join(f"f{k}={v:.3f}({v * 255:.0f})" for k, v in entries)
        report = (f"frames={f.shape[0]} | keyframes: {detail} | "
                  f"f range {float(f.min()):.3f}..{float(f.max()):.3f} | interp={interpolation}")
        return ({"focus": f, "n": int(f.shape[0])}, curve, report)


NODE_CLASS_MAPPINGS = {
    "AnyToBokehDisparity": AnyToBokehDisparity,
    "AnyToBokehFocusFromMask": AnyToBokehFocusFromMask,
    "AnyToBokehFocusControl": AnyToBokehFocusControl,
    "AnyToBokehLoadModel": AnyToBokehLoadModel,
    "AnyToBokehPreviewCoC": AnyToBokehPreviewCoC,
    "AnyToBokehGenerate": AnyToBokehGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyToBokehDisparity": "Any-to-Bokeh: Disparity from Depth",
    "AnyToBokehFocusFromMask": "Any-to-Bokeh: Focus Plane from Mask",
    "AnyToBokehFocusControl": "Any-to-Bokeh: Focus Control (curve / advanced text)",
    "AnyToBokehLoadModel": "Any-to-Bokeh: Load Model",
    "AnyToBokehPreviewCoC": "Any-to-Bokeh: Preview CoC / MPI",
    "AnyToBokehGenerate": "Any-to-Bokeh: Generate (Video Refocus)",
}
