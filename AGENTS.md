# ComfyUI-AnyToBokeh_v2

Video **refocusing** for ComfyUI: you keep one sharp master clip, choose the focal plane, and the
model renders the bokeh. Rack focus becomes a deterministic re-render instead of a regeneration, so
the subject cannot drift between takes.

Wraps **[vivoCameraResearch/any-to-bokeh](https://github.com/vivoCameraResearch/any-to-bokeh)**
(ICLR 2026) — one-step video diffusion on a modified SVD-XT backbone conditioned on a CoC map, a
focal-plane-adapted MPI mask and a blur strength `k`.
Paper: [arXiv:2505.21593](https://arxiv.org/abs/2505.21593)

Depth and masks are **not** reimplemented here — they come from ComfyUI **core** nodes
(`LoadDA3Model` + `DA3Inference`, `CheckpointLoaderSimple` + `CLIPTextEncode` + `SAM3_Detect`).
This pack only builds the tensors the model expects and runs it.

> **v2 = the slim release.** It contains exactly the six nodes the two workflows in
> [`workflows/`](workflows) use, and nothing else. If you want the extra focus-authoring nodes
> (`FocusSchedule`, `FocusRows`, `LoadDisparityFolder`), see
> [ComfyUI-AnyToBokeh](https://github.com/alexanz314/ComfyUI-AnyToBokeh) (v1) — but **do not enable
> both packs at once**: the node class names are identical, so ComfyUI keeps whichever pack it loads
> first and logs duplicate-name warnings for the other.

> **License: CC BY-NC 4.0 — non-commercial use only.** The vendored model code in
> [`any2bokeh_repo/`](any2bokeh_repo) is upstream's, under the same licence, and the weights are
> fine-tunes of Stability AI's SVD-XT. See [License](#license).

---

## Contents

- [What the two workflows do](#what-the-two-workflows-do)
- [Install](#install)
- [Weights](#weights)
- [Running a workflow](#running-a-workflow)
- [Node reference](#node-reference)
- [Authoring the focal plane](#authoring-the-focal-plane)
- [Settings that matter](#settings-that-matter)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Tests and verification status](#tests-and-verification-status)
- [What came from upstream](#what-came-from-upstream)
- [License](#license)

---

## Repository layout

```
ComfyUI-AnyToBokeh_v2/
├── __init__.py                  node registration + WEB_DIRECTORY
├── nodes.py                     the six ComfyUI nodes
├── a2b/prepare.py               ported preprocessing (pure torch/numpy, no diffusers)
├── runner/run_any2bokeh.py      model runner, executed as a subprocess
├── any2bokeh_repo/              vendored upstream model code + its LICENSE.txt
│   ├── a2b_compat.py            diffusers 0.25 -> 0.39 compatibility shim
│   ├── a2b_models/              unet / vae / attention / blocks
│   └── a2b_pipelines/           the Any-to-Bokeh SVD pipeline
├── web/js/focus_curve_editor.js draggable focal-plane curve on the FocusControl node
├── workflows/                   the two published graphs (drag & drop into ComfyUI)
├── scripts/download_weights.sh  fetches every weight except the Google-Drive fine-tune
└── tests/                       numeric, structural and smoke tests (no weights needed)
```

---

## What the two workflows do

Both graphs share the same backbone and differ only in **where the focal plane comes from**.

```
LoadVideo ─► GetVideoComponents ─┬─────────────────────────────────────────────┐
                                 │ IMAGE (the sharp master clip)                │
LoadDA3Model ─► DA3Inference ────┴─► DA3_GEOMETRY                               │
                                            │                                  │
                              Any-to-Bokeh: Disparity from Depth                │
                                            │ A2B_DISPARITY                    │
                       ┌────────────────────┴─────────────────────┐            │
                       ▼ (curve workflow)                         ▼ (sam workflow)
     Any-to-Bokeh: Focus Control              CheckpointLoaderSimple(sam3.1)
     (draggable curve / keyframes)            └► CLIPTextEncode("person")
                       │                         └► SAM3_Detect ─► MASK
                       │                                └► Any-to-Bokeh: Focus Plane from Mask
                       └────────────┬───────────────────────────────┘
                                    ▼ A2B_FOCUS
   Any-to-Bokeh: Load Model ─► Any-to-Bokeh: Generate ─► IMAGE ─┐
                                    │                          ├─► CreateVideo ─► SaveVideo
   (Previews: disparity, CoC, MPI regions, mask, curve)         │
                                          AddLabel + ImageConcatMulti (side-by-side compare)
```

### 1. `workflows/any2bokeh_curve_00001_.json` — manual rack focus

You write the focal-plane scenario yourself. Shipped settings: 240-frame clip, keyframes
`0:0.75;75:0.5;200:0.0` with **linear** interpolation (near → mid → far, i.e. a classic pull-focus
off the subject), `k=16`, 576×1024, `mpi_alpha=1.5`, `num_inference_steps=1`, `motion_bucket_id=127`,
`fps=7`, `noise_aug_strength=0`. The curve is drawn on the node and can be dragged before rendering.

### 2. `workflows/any2bokeh_sam_mask_00001_.json` — focus follows a subject

SAM3.1 detects `person` (change the text to any noun phrase), and the focal plane per frame becomes
the **mean disparity inside that mask** — upstream's `disp_concept`. The subject stays sharp while
everything else falls out of focus, and the plane follows the subject as it moves. Shipped settings:
`threshold=0.3`, `refine_iterations=2`, `individual_masks=false`, same `k`/resolution as above.

Both graphs also wire `Any-to-Bokeh: Preview CoC / MPI` and preview nodes for disparity, CoC, MPI
regions and the mask — **those run without any model weights**, so you can validate a graph before
downloading 11 GB.

---

## Install

### Requirements

| | |
|---|---|
| ComfyUI | **≥ 0.37.0** — `LoadDA3Model`/`DA3Inference` and `SAM3_Detect` are core nodes added recently |
| Python | ≥ 3.10 (tested on 3.12) |
| GPU | 24 GB VRAM is comfortable at 576×1024. See [Known limitations](#known-limitations) for smaller cards |
| Custom nodes | [comfyui-kjnodes](https://github.com/kijai/ComfyUI-KJNodes) — only for the optional `AddLabel` / `ImageConcatMulti` side-by-side compare in the shipped graphs. Delete those three nodes if you do not want the dependency |
| Frontend | `MarkdownNote` (the documentation node in the graphs) is provided by the ComfyUI web frontend itself, not by a Python pack |

Python packages are listed in [`requirements.txt`](requirements.txt): `torch`, `numpy`,
`diffusers>=0.35`, `transformers>=4.40`, `einops`, `Pillow`, `huggingface_hub`. A stock ComfyUI venv
already satisfies `torch`/`numpy`; the rest is small.

> **Do not** install upstream's pins (`diffusers==0.25.0`, `transformers==4.32.1`) into the ComfyUI
> venv — that breaks unrelated custom nodes. [`any2bokeh_repo/a2b_compat.py`](any2bokeh_repo/a2b_compat.py)
> aliases the module paths that moved since 0.25, so the vendored code runs unmodified on the modern
> stack (verified on diffusers 0.39.0 / transformers 5.13.1).

### Option A — git clone

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/alexanz314/ComfyUI-AnyToBokeh_v2.git
cd ComfyUI-AnyToBokeh_v2
<ComfyUI>/venv/bin/pip install -r requirements.txt
```

Restart ComfyUI. The nodes appear under the category **AnyToBokeh**
(`Any-to-Bokeh: Disparity from Depth`, `… Generate (Video Refocus)`, …).

### Option B — ComfyUI Manager

Search for **AnyToBokeh** and install `ComfyUI-AnyToBokeh_v2`, then restart.

### Option C — comfy-cli / Comfy Registry

```bash
comfy install-node comfyui-anytobokeh-v2
```

[`pyproject.toml`](pyproject.toml) carries the `[tool.comfy]` block the registry reads
(`custom_nodes = "nodes.py"`, `requirements = "requirements.txt"`, empty `class_prefix` — the class
names are part of the shipped workflows' contract).

---

## Weights

**Nothing is bundled.** ~11.4 GB total:

| # | Path (under `<ComfyUI>/models/`) | Size | Source |
|---|---|---|---|
| 1 | `any2bokeh/unet/` | 7.0 GB | upstream Google Drive — see the [upstream README](https://github.com/vivoCameraResearch/any-to-bokeh) |
| 2 | `any2bokeh/vae/` | 375 MB | same archive |
| 3 | `any2bokeh/svd_xt/` | 1.2 GB | [`stabilityai/stable-video-diffusion-img2vid-xt`](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) — **slice only**, see below |
| 4 | `geometry_estimation/depth_anything_3_mono_large.safetensors` | 1.24 GB | [`Comfy-Org/Depth-Anything-3`](https://huggingface.co/Comfy-Org/Depth-Anything-3) |
| 5 | `checkpoints/sam3.1_multiplex_fp16.safetensors` | 1.63 GB | [`Comfy-Org/sam3.1`](https://huggingface.co/Comfy-Org/sam3.1) — only the SAM-mask workflow |

All three HuggingFace repos are **public** (checked 2026-10-31: no gate on SVD-XT, DA3 or sam3.1), so
no token is required.

### One command

```bash
bash <ComfyUI>/custom_nodes/ComfyUI-AnyToBokeh_v2/scripts/download_weights.sh
```

It skips anything already on disk, never overwrites, and installs nothing into your environment
unless you pass `INSTALL_GDOWN=1` (Google Drive needs `gdown`). Useful overrides:

```bash
COMFY_MODELS=/data/comfy/models SKIP_DRIVE=1 bash scripts/download_weights.sh   # weights elsewhere
PY=/opt/comfy/venv/bin/python   bash scripts/download_weights.sh               # another interpreter
SKIP_SAM=1 bash scripts/download_weights.sh                                    # curve workflow only
```

### SVD-XT: fetch the slice, not the repo

The full repo is 21.5 GiB; diffusers needs ~1.2 GiB of it. `unet/` and `vae/` weights are replaced by
the Any-to-Bokeh fine-tune (only their `config.json` is kept), the single-file
`svd_xt*.safetensors` checkpoints are for ComfyUI's *native* loader, and upstream loads with
`variant="fp16"`, so only `image_encoder/model.fp16.safetensors` is used:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(
  repo_id='stabilityai/stable-video-diffusion-img2vid-xt',
  local_dir='<ComfyUI>/models/any2bokeh/svd_xt',
  allow_patterns=['model_index.json','LICENSE.md','scheduler/*','feature_extractor/*',
                  'image_encoder/config.json','image_encoder/model.fp16.safetensors',
                  'unet/config.json','vae/config.json'])"
```

### Items 1–2 by hand

The fine-tune is distributed as a Google Drive archive. Download it from the link in the upstream
README and unpack so that these exist (diffusers layout — **directories**, not loose files):

```
<ComfyUI>/models/any2bokeh/unet/   config.json + diffusion_pytorch_model.safetensors
<ComfyUI>/models/any2bokeh/vae/    config.json + diffusion_pytorch_model.safetensors
<ComfyUI>/models/any2bokeh/svd_xt/ model_index.json + scheduler/ + feature_extractor/ + image_encoder/
```

Keeping them in a dedicated `models/any2bokeh/` subtree is deliberate: `models/unet/` and
`models/vae/` are listed by ComfyUI's own `UNETLoader`/`VAELoader`, and a bare 7 GB safetensors there
would show up in those pickers and fail to load.

Weights somewhere else entirely? Either type absolute paths into the **Load Model** node, or
`export ANY2BOKEH_MODELS=/mnt/models/any2bokeh:/other/place` (colon-separated) — the loader reports
every directory it searched when a path does not resolve.

---

## Running a workflow

1. Drag a JSON from [`workflows/`](workflows) onto the ComfyUI canvas.
2. **Fix the paths.** The shipped graphs were saved on the author's machine, so three widgets hold
   her absolute paths — replace them with yours (or with the relative forms, which resolve against
   `<ComfyUI>/models/`):

   | Node | Widget | Shipped value | Use instead |
   |---|---|---|---|
   | `Any-to-Bokeh: Load Model` | `unet_dir` | `/home/user/work/ComfyUI/models/any2bokeh/unet` | `any2bokeh/unet` |
   | | `vae_dir` | `/home/user/work/ComfyUI/models/any2bokeh/vae` | `any2bokeh/vae` |
   | | `svd_dir` | `/home/user/work/ComfyUI/models/any2bokeh/svd_xt` | `any2bokeh/svd_xt` |
   | `Any-to-Bokeh: Generate` | `python_bin` | `/home/user/work/ComfyUI/venv/bin/python` | *empty* (auto-detect) |

3. **Point `LoadVideo` at your own clip.** Both graphs reference `a2b_demo_Dh3EhO6vbY_69.mp4`, the
   upstream demo clip, which is not redistributed here.
4. **Check the graph without weights first.** Run only the preview branch: `Disparity from Depth` →
   `Focus Control` (or `Focus Plane from Mask`) → `Preview CoC / MPI`. If the disparity preview, the
   CoC map and (for the SAM graph) the mask preview look right, the render will too.
5. Queue the full graph. Output goes to `output/video/any2bokeh_curve_*.mp4` /
   `any2bokeh_sam_mask_*.mp4`; the two `AddLabel` + `ImageConcatMulti` nodes make it a side-by-side
   "Original | Any-to-Bokeh" compare.

The `MarkdownNote` in each graph carries the same weight table as a reminder.

---

## Node reference

All six live under the category **AnyToBokeh**.

| Node (class name) | Inputs → outputs | What it does |
|---|---|---|
| **Any-to-Bokeh: Disparity from Depth** (`AnyToBokehDisparity`) | `DA3_GEOMETRY` *or* depth `IMAGE` → `A2B_DISPARITY`, `preview` | Converts depth to the clip-globally normalised `uint8` disparity upstream consumes (single min/max over the **whole** clip — that is what keeps it temporally stable). The `convention` combo handles the sign: DA3/MoGe depth is *larger = farther* and gets inverted; Video-Depth-Anything / DAv2 output is already disparity-like. Prefer the raw `DA3_GEOMETRY` over `DA3Render`, which is normalised and colormapped. |
| **Any-to-Bokeh: Focus Control** (`AnyToBokehFocusControl`) | `A2B_DISPARITY`, `keyframes`, `interpolation` → `A2B_FOCUS`, `curve`, `report` | The manual scenario: `frame:target` pairs, drawn as a draggable curve on the node. One entry = fixed focal plane for the whole clip; several = a rack focus with `linear` / `smooth` (cosine ease) / `hold` interpolation. See [Authoring the focal plane](#authoring-the-focal-plane). |
| **Any-to-Bokeh: Focus Plane from Mask** (`AnyToBokehFocusFromMask`) | `A2B_DISPARITY`, `MASK` → `A2B_FOCUS`, `mask_preview`, `report` | Per-frame plane = mean disparity inside the mask (upstream `disp_concept`). Frames with an empty mask **carry** the plane from the nearest detection instead of silently defaulting; if the mask is empty on *every* frame the node raises rather than blurring everything. The report prints focus min/mean/max, mask coverage and how many frames were carried. |
| **Any-to-Bokeh: Load Model** (`AnyToBokehLoadModel`) | `unet_dir`, `vae_dir`, `svd_dir`, `dtype`, `svd_variant` → `A2B_PIPE` | Resolves and validates the three diffusers-format directories (they are folders, not single files). Cheap: it loads nothing, it only checks the layout, so put it early in the graph and get path errors before a long render. |
| **Any-to-Bokeh: Preview CoC / MPI** (`AnyToBokehPreviewCoC`) | `A2B_DISPARITY`, `A2B_FOCUS`, `k`, `mpi_alpha` → `coc`, `mpi_regions`, `report` | Debug view of the two conditioning maps actually fed to the model. **Needs no weights** — use it to validate a scenario. CoC: white = far from the focal plane = blurred. |
| **Any-to-Bokeh: Generate** (`AnyToBokehGenerate`) | `A2B_PIPE`, `IMAGE`, `A2B_DISPARITY`, `A2B_FOCUS`, + settings → `IMAGE`, `coc_preview`, `report` | Runs the model in a subprocess and returns the refocused clip. The report prints `frames_in`, `frames_out`, group count, `k` range and size. |

Custom socket types: `A2B_DISPARITY` (clip-normalised `uint8` disparity), `A2B_FOCUS` (per-frame
plane in `[0,1]`, `1.0` = nearest), `A2B_PIPE` (resolved weight paths + dtype).

---

## Authoring the focal plane

Values are disparity units: **`1.0` (or `255`) = nearest, `0.0` = farthest.**

`keyframes` accepts `frame:target` pairs separated by `;` or newlines. `frame` is an index or the
word `end`; `target` can be written four ways:

| Form | Meaning | Example |
|---|---|---|
| `0.42` | explicit plane, 0..1 | `0:0.75` |
| `180` | explicit plane in 0..255 (upstream's `_zf_` convention) | `120:139` |
| `near` / `mid` / `far` | 90th / 50th / 10th percentile of **that frame's** disparity | `0:near; end:far` |
| `0.45,0.62` | a **point** in 0..1 frame coordinates: median disparity in a small window around it — "focus on whatever is under this pixel" | `60:0.5,0.4` |

Values before the first key and after the last one are **held**. One key only = a fixed plane for the
whole clip.

**The curve editor** (installed automatically from [`web/js/`](web/js)) draws the scenario straight
on the node:

- **click** empty space → add a keyframe there
- **drag** a point → move it (the tooltip shows `frame` and `plane`, both in 0..1 and /255)
- **right-click** a point → delete it
- every edit is written back into the `keyframes` widget, so the saved workflow stays the single
  source of truth

`clip_frames` only sets the x-axis length of that drawing (0 = auto); the server always uses the real
clip length from the connected disparity.

Typical scenarios:

| Goal | Keyframes |
|---|---|
| Subject sharp, background blurred (fixed) | `0:0.75` |
| Pull focus subject → background | `0:0.75; end:0.1` |
| Upstream demo ramp (near → mid → far) | `0:0.9;24:0.1` style, or `0:0.75;75:0.5;200:0.0` as shipped |
| Focus on whatever is at the centre | `0:0.5,0.5` |

---

## Settings that matter

| Widget | Default | Notes |
|---|---|---|
| `k` | 16 | Blur strength; upstream's demo CSV uses 16. `k_per_frame` overrides it per frame (`"8,16,24"` or one value per frame, linearly interpolated) |
| `mpi_alpha` | 1.5 | Shapes the MPI region thresholds. Upstream uses 1.5 for inference, 1.0 for training |
| `height` × `width` | 576 × 1024 | What the model was trained on. Other sizes run, but are off-distribution |
| `group_frames` / `overlap_frames` | 8 / 4 | Upstream's temporal window and cross-fade. `overlap_frames` must be `< group_frames` |
| `infer_clip_num` | 2 | **Must stay 2** — upstream supports nothing else, and the node rejects other values |
| `num_inference_steps` | 1 | Upstream is a single deterministic forward pass |
| `motion_bucket_id` / `fps` / `noise_aug_strength` | 127 / 7 / 0.0 | Upstream's SVD conditioning values |
| `seed` | 0 | With `num_inference_steps=1` and `noise_aug_strength=0` the pass is near-deterministic |
| `backend` | subprocess | `subprocess` runs the model in a separate interpreter, keeping ~7.5 GB of weights out of ComfyUI's allocator (it also calls `unload_all_models()` first). `in-process` exists for machines whose venv already satisfies upstream's pins |
| `python_bin` | *empty* | Interpreter for the subprocess. Empty = a `venv/`/`.venv/` inside this pack if present, else the interpreter ComfyUI runs on |
| `timeout_minutes` | 120 | Raise it for long clips on small cards |

Environment variables:

| Var | Effect |
|---|---|
| `ANY2BOKEH_MODELS` | extra colon-separated directories searched for the weight folders |
| `A2B_NO_OFFLOAD=1` | disable `enable_model_cpu_offload()` — faster if you have VRAM to spare |
| `A2B_DEBUG=1` | print what `a2b_compat` aliased/stubbed |

---

## Known limitations

- **Minimum clip length.** Upstream's `decode_latents` fuses the overlap of the *first* batch by
  reading `latents[1:2]` unconditionally, and only appends the tail frames when a `latent_cache`
  already exists — so the first batch must not also be the last one. With `infer_clip_num=2` that
  means **at least 3 groups**, i.e. **≥ 13 frames** at `group_frames=8` / `overlap_frames=4`
  (generally `overlap_frames + 2*(group_frames-overlap_frames) + 1`). Shorter clips are rejected by
  the Generate node with a message saying exactly how many frames are needed; upstream would have
  died inside `torch.cat` with `Expected size 1 but got size 0`.
- **Frame count can differ from the input count** when the last group is padded; the report string
  prints both numbers.
- **VRAM.** At 576×1024 the pipeline holds the UNet (~3.8 GB), the CLIP image encoder (~1.2 GB) and
  the VAE. The runner enables `enable_model_cpu_offload()` and decodes 2 frames at a time
  (`decode_chunk_size=2`) specifically so a 24 GB card does not OOM during the VAE decode. If you
  still OOM: lower the resolution, keep offload on, and make sure nothing else holds VRAM.
- **`in-process` backend** will fail on a venv with upstream's pins missing; it is not the default.
- **No training code.** Inference only.
- **Depth distribution.** The model was trained on Video-Depth-Anything disparity. DA3 works (both
  shipped graphs use it), but if the bokeh looks wrong on your footage, try VDA
  (`video_depth_anything_vits.pth`) through the `depth_image` input with
  `convention = disparity: larger = closer`.
- **Duplicate node names with v1.** See the note at the top.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no directory at … (also searched: […])` | Weights are not where the widget says. Use paths relative to `<ComfyUI>/models/` (`any2bokeh/unet`) or set `ANY2BOKEH_MODELS` |
| `… has no model_index.json` | `svd_dir` is not the diffusers layout of SVD-XT. Run `scripts/download_weights.sh` |
| `clip too short for upstream's overlap blending` | See [Known limitations](#known-limitations): ≥ 13 frames at the default grouping |
| `the focus-subject mask is empty on EVERY frame` | SAM detected nothing: check the text prompt, lower `threshold`, and look at the mask preview node |
| `Expected size 1 but got size 0` inside `torch.cat` | You are on v1 without the short-clip guard, and the clip yields one group. Lengthen the clip |
| Runner fails with `import error: No module named 'diffusers.models.transformer_2d'` | `a2b_compat` did not run or diffusers is older than 0.35. `pip install -r requirements.txt` |
| `RuntimeError: Input type (c10::Half) and bias type (float) should be the same` | The SVD VAE's `force_upcast=True`; the runner clears it and casts the parameters. If you see this, your `svd_variant`/`dtype` combination is unusual — try `dtype=float16` + `svd_variant=fp16` |
| CUDA OOM during decode | Keep `enable_model_cpu_offload` (do **not** set `A2B_NO_OFFLOAD=1`), lower the resolution, close other GPU work |
| Red `AddLabel` / `ImageConcatMulti` nodes | Install [comfyui-kjnodes](https://github.com/kijai/ComfyUI-KJNodes), or delete those nodes and wire Generate → CreateVideo directly |
| Red `MarkdownNote` | Frontend-provided; harmless — delete it |

---

## Tests and verification status

```bash
cd <ComfyUI>/custom_nodes/ComfyUI-AnyToBokeh_v2

python tests/test_prepare.py            # 10 numeric checks, torch+numpy only
python tests/test_workflows.py          # both shipped graphs: links, slots, types, node coverage
<ComfyUI>/venv/bin/python tests/test_nodes_smoke.py
                                        # registration, chains, widget names, runner error path
A2B_RUN_MODEL=1 <ComfyUI>/venv/bin/python tests/test_nodes_smoke.py
                                        # ... plus one real render (needs weights + GPU)
A2B_RUN_MODEL=1 A2B_RENDER_FRAMES=13 ... # render at the exact minimum clip length
```

Each file is a plain script (no pytest needed); they are also written to be pytest-collectable —
`pytest -q tests/` picks the checks up, `test_nodes_smoke.py` gets the pack from a module fixture and
skips the real render unless `A2B_RUN_MODEL=1`.

**Verified on** Linux, NVIDIA RTX 4090 (24 GB), ComfyUI 0.37.0, Python 3.12, torch 2.13.0+cu130,
diffusers 0.39.0, transformers 5.13.1, einops 0.8.2, numpy 2.4.6 — every number below is a real run
of these commands, not a claim:

- `tests/test_prepare.py` — **10/10 passed**: clip-global `uint8` normalisation and depth inversion,
  mask focal plane (exact `disp_concept` value, empty-frame carry-forward, all-empty raises),
  MPI mask range, `k` parsing, frame grouping and cosine overlap weights summing to 1.0,
  `build_groups` shapes/ranges/padding, all four keyframe target forms, fixed vs. scenario
  interpolation (`linear`/`smooth`/`hold`), the shipped curve `0:0.75;75:0.5;200:0.0` on a 240-frame
  clip, curve image rendering.
- `tests/test_workflows.py` — **2/2 passed**: 18 nodes/21 links and 22 nodes/27 links, every link
  agrees with the slots it connects (union types like `IMAGE,MASK` handled), no required input of a
  pack node left dangling, and the set of registered nodes is **exactly** the set the two graphs use.
- `tests/test_nodes_smoke.py` — **7/7 passed** (8/8 with `A2B_RUN_MODEL=1`): node registration,
  workflow widget names vs. `INPUT_TYPES`, both focus chains on synthetic tensors, `LoadModel` path
  validation, the runner's one-line JSON error path, short-clip rejection.
- **Real renders through the model**: 16 frames → 16 frames (3 groups) and the 13-frame boundary
  → 13 frames, both at 576×1024, `k=16`, `num_inference_steps=1`. Both shipped graphs have also been
  run end-to-end on a 240-frame clip (their outputs are the `_00001_` names the JSONs were saved from).

**Not verified:** multi-GPU; Windows/macOS; cards below 24 GB; resolutions other than 576×1024;
bit-exact equality with upstream's own `test/inference_demo.py` output (the port is line-by-line
faithful and numerically tested, but not diffed against their script's frames); collection under
`pytest` (pytest is not installed in the environment above, so the files were run as the plain
scripts they are). Two deliberate deviations: `mpi_mask` is resized with
`torch.nn.functional.interpolate(bilinear, align_corners=False)` instead of `torchvision…resize`
(same default behaviour, not bit-identical), and disparity stays in memory instead of round-tripping
through PNG (the `uint8` truncation is reproduced).

---

## What came from upstream

| This pack | Upstream source |
|---|---|
| `a2b/prepare.py :: gen_mpi_mask` | `utils/dataset.py :: gen_mpi_mask` (verbatim) |
| `depth_to_disparity` | `utils/pre_process.py` — clip-global min/max → ×255 → `uint8` truncation |
| `focus_from_mask` | `utils/pre_process.py :: disp_concept` — `disp_norm[binary_mask].mean() / 255` |
| `group_frames_with_overlap` | `utils/dataset.py :: BokehCocMPIDatasetWholeVid.group_frames_with_overlap` (cosine weights, two-pass weight resolution) |
| `build_groups` | `utils/dataset.py :: get_batch` — CoC `|disp − f·255|/255`, MPI `alpha=1.5`, `Resize(576,1024)` + `Normalize(.5,.5)`, zero-padding of the last group |
| `runner/run_any2bokeh.py` | `test/inference_demo.py` — `infer_clip_num=2`, `latent_cache` carried across clips, `is_last_batch`, `batch_size=2 if weight.shape[0]==infer_clip_num else 1`, `num_inference_steps=1`, `motion_bucket_id=127`, `fps=7`, `noise_aug_strength=0.0` |
| `any2bokeh_repo/` | their `models/` and `pipelines/`, renamed to `a2b_models/` / `a2b_pipelines/` so they cannot clash with ComfyUI's own `models/` package (one import line patched in `a2b_pipelines/any2bokeh_pipe.py`). Their `LICENSE.txt` is kept alongside |
| `any2bokeh_repo/a2b_compat.py` | **new in this pack** — aliases/stubs for the diffusers modules that moved or were removed since the 0.25 pin, so the vendored files stay unmodified and upstream updates remain mergeable |

Additions of this pack on top of a straight port: the interactive curve editor, the `A2B_*` socket
types, the short-clip guard, `decode_chunk_size=2` and `enable_model_cpu_offload()` for 24 GB cards,
the `force_upcast` fix for the SVD VAE on modern diffusers, and carrying (rather than defaulting) the
focal plane across frames whose mask is empty.

---

## License

**Creative Commons Attribution-NonCommercial 4.0 International** — see [`LICENSE`](LICENSE).
**No commercial use.**

Why the whole repository carries it: [`any2bokeh_repo/`](any2bokeh_repo) is upstream code, and
upstream's [`LICENSE.txt`](any2bokeh_repo/LICENSE.txt) is CC BY-NC 4.0 ("Non-commercial use is freely
permitted"). `a2b/prepare.py` and `runner/run_any2bokeh.py` are ports of that same code, so they
inherit the terms. Keep both licence files in place if you redistribute.

The weights are a separate matter: the archive ships no licence file, but both `config.json` files
declare `_name_or_path: stabilityai/stable-video-diffusion-img2vid-xt`, so these are **fine-tunes of
SVD-XT** and Stability AI's SVD Community License likely applies to the derivative as well. Check
before any redistribution of weights.

Attribution: Any-to-Bokeh by vivo Camera Research — https://github.com/vivoCameraResearch/any-to-bokeh ·
[arXiv:2505.21593](https://arxiv.org/abs/2505.21593) (ICLR 2026). Depth Anything 3 and SAM3.1
checkpoints come from Comfy-Org's HuggingFace organisations and carry their own licences.
