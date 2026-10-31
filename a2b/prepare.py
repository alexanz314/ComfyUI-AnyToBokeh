"""Ported preprocessing logic for Any-to-Bokeh (vivoCameraResearch/any-to-bokeh, ICLR 2026).

Everything here runs inside the ComfyUI venv and needs only torch + numpy: no diffusers,
no transformers. The heavy model call is delegated to runner/run_any2bokeh.py.

Each function mirrors the upstream implementation it replaces, so results are comparable:

  depth_to_disparity      <- utils/pre_process.py :: img_synthsis()
  focus_from_mask         <- utils/pre_process.py :: disp_concept
  group_frames_with_overlap <- utils/dataset.py :: BokehCocMPIDatasetWholeVid.group_frames_with_overlap()
  build_groups            <- utils/dataset.py :: BokehCocMPIDatasetWholeVid.get_batch()
  gen_mpi_mask            <- utils/dataset.py :: gen_mpi_mask()  (copied verbatim)

This module is the subset the two published workflows need (curve-driven focus and mask-driven
focus). Upstream's three-point `customize_f.py` schedule is not part of it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as TF

# Upstream inference constants (test/inference_demo.py).
DEFAULT_HEIGHT = 576
DEFAULT_WIDTH = 1024
DEFAULT_GROUP_FRAMES = 8
DEFAULT_OVERLAP_FRAMES = 4
DEFAULT_INFER_CLIP_NUM = 2  # upstream: "Only support 2"
MPI_SEGMENTS = 4
MPI_ALPHA_INFER = 1.5  # dataset.get_batch (validation); training uses 1.0


def gen_mpi_mask(D: torch.Tensor, f: float, segments: int = MPI_SEGMENTS, alpha: float = 2.0) -> torch.Tensor:
    """Verbatim copy of utils/dataset.py :: gen_mpi_mask."""
    diff = torch.abs(D - f)
    sharpness = f ** alpha if f > 0 else 1e-6
    base = torch.linspace(0, 1, segments + 1, device=D.device)
    thresholds = base ** (1 / sharpness)
    region_mask = torch.bucketize(diff, thresholds[1:-1])
    return region_mask


def depth_to_disparity(depth: torch.Tensor, larger_is_farther: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """(N,H,W) float depth -> (N,H,W) uint8 disparity normalised over the WHOLE clip.

    Upstream (utils/pre_process.py) takes Video-Depth-Anything output, treats it directly as
    disparity (`disparitys = depths`), then normalises with a single clip-global min/max:

        d_min, d_max = disparitys.min(), disparitys.max()      # over all frames
        disp_norm = ((disp - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    Clip-global (not per-frame) normalisation is what keeps the result temporally stable, so it
    is reproduced exactly. `.astype(np.uint8)` truncates, hence `.to(torch.uint8)` and not round().

    Depth Anything 3 returns depth in the MoGe convention (larger = farther), which is the
    opposite of the disparity convention upstream expects, so it is inverted here.
    """
    if depth.ndim != 3:
        raise ValueError(f"depth must be (N,H,W), got {tuple(depth.shape)}")
    d = depth.float()
    if larger_is_farther:
        d = 1.0 / d.clamp_min(eps)
    if not torch.isfinite(d).all():
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d_min, d_max = d.min(), d.max()
    denom = (d_max - d_min).clamp_min(1e-8)
    norm = (d - d_min) / denom * 255.0
    return norm.clamp(0.0, 255.0).to(torch.uint8)


def focus_from_mask(disp_u8: torch.Tensor, mask: torch.Tensor, fallback: float | None = None) -> tuple[torch.Tensor, list[int]]:
    """Per-frame focal plane in [0,1], derived from the focus-subject mask.

    Upstream (utils/pre_process.py):

        _, binary = cv2.threshold(mask, 128, 1, THRESH_BINARY)
        disp_concept = disp_norm[binary].mean() / 255

    Frames with an empty mask are skipped upstream (the file is not written at all). Here they are
    carried forward from the last detected plane (leading frames are back-filled from the first
    detection), because a silent mid-range fallback blurs subject AND background and reads as
    "the model is broken". If the mask is empty on every frame this raises instead.

    Returns (focus (N,), indices of frames whose mask was empty).
    """
    if disp_u8.shape[0] != mask.shape[0]:
        raise ValueError(f"disparity has {disp_u8.shape[0]} frames but mask has {mask.shape[0]}")
    d = disp_u8.float()
    m = (mask > 0.5).to(d.device)
    if m.shape[1:] != d.shape[1:]:
        m = TF.interpolate(m.unsqueeze(1).float(), size=d.shape[1:], mode="nearest").squeeze(1) > 0.5
    focus = torch.full((d.shape[0],), float("nan"))
    empty: list[int] = []
    for i in range(d.shape[0]):
        sel = d[i][m[i]]
        if sel.numel() == 0:
            empty.append(i)
            continue
        focus[i] = sel.mean() / 255.0
    if empty:
        if not torch.isfinite(focus).any():
            raise ValueError(
                "the focus-subject mask is empty on EVERY frame: nothing was detected. A mid-range "
                "fallback plane would blur both subject and background, which reads as 'the model is "
                "broken', so this is a hard error instead. Check the subject text / detection "
                "threshold, or look at the mask preview."
            )
        idx = torch.arange(focus.shape[0])
        last_valid = torch.cummax(torch.where(torch.isnan(focus), -1, idx), 0)[1]
        focus = torch.where(torch.isnan(focus), focus[last_valid], focus)
        first_valid = int((~torch.isnan(focus)).float().argmax())
        focus[:first_valid] = focus[first_valid]
        if fallback is not None:
            focus = torch.nan_to_num(focus, nan=float(fallback))
    return focus, empty


def parse_k_list(text: str, n_frames: int, default_k: float) -> torch.Tensor:
    """Per-frame blur strength k. Empty string -> constant `default_k`.

    Mirrors upstream, where k is either a single CSV value or read per frame from `_k_` in the
    disparity filename (utils/customize_k.py).
    """
    text = (text or "").strip()
    if not text:
        return torch.full((n_frames,), float(default_k))
    vals = [float(p) for p in text.split(",") if p.strip()]
    if len(vals) == 1:
        return torch.full((n_frames,), vals[0])
    if len(vals) == n_frames:
        return torch.tensor(vals)
    idx = torch.arange(n_frames, dtype=torch.float32)
    k_t = torch.linspace(0, n_frames - 1, len(vals))
    v_t = torch.tensor(vals, dtype=torch.float32)
    pos = torch.searchsorted(k_t.contiguous(), idx.contiguous(), right=True).clamp(1, len(vals) - 1)
    k0, k1 = k_t[pos - 1], k_t[pos]
    v0, v1 = v_t[pos - 1], v_t[pos]
    t = ((idx - k0) / (k1 - k0).clamp_min(1e-6)).clamp(0.0, 1.0)
    return v0 + (v1 - v0) * t


def _cosine_weight(position: int, overlap: int) -> float:
    """utils/dataset.py :: get_batch.cosine_weight"""
    return 0.5 * (1 + math.cos(math.pi * position / overlap))


def group_frames_with_overlap(n_frames: int, group_frames: int, overlap_frames: int) -> tuple[list[list[int]], list[list[float]]]:
    """Port of BokehCocMPIDatasetWholeVid.group_frames_with_overlap with the cosine weight fn.

    Upstream keys weights by filename, and a frame belonging to two groups ends up with the value
    written by the later group; the earlier group then uses `1 - weights[file]` for its overlap
    slots. The same two-pass behaviour is reproduced here on frame indices so the blending matches.
    """
    files = list(range(n_frames))
    if n_frames <= group_frames:
        return [files], [[1.0] * n_frames]

    weights: dict[int, float] = {}
    groups: list[list[int]] = []
    step = group_frames - overlap_frames
    num_groups = (n_frames - overlap_frames) // step
    if (n_frames - overlap_frames) % step != 0:
        num_groups += 1

    for i in range(num_groups):
        start = i * step
        end = min(start + group_frames, n_frames)
        group = files[start:end]
        groups.append(group)
        for j, f in enumerate(group):
            if i > 0 and j < overlap_frames:
                weights[f] = _cosine_weight(j, overlap_frames)
            else:
                weights[f] = 1.0

    group_weights: list[list[float]] = []
    for i, group in enumerate(groups):
        gw = []
        for j, f in enumerate(group):
            if i > 0 and j < overlap_frames:
                gw.append(1 - weights.get(f, 1.0))
            else:
                gw.append(weights.get(f, 1.0))
        group_weights.append(gw)
    return groups, group_weights


def _resize_norm(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Equivalent of upstream pixel_transforms: Resize((H,W)) + CenterCrop((H,W)) + Normalize(.5,.5).

    Resize already forces the target size, so CenterCrop is a no-op and is omitted. Input (F,C,H,W)
    in [0,1]; output (F,C,H,W) in [-1,1].
    """
    if x.shape[-2:] != size:
        x = TF.interpolate(x.float(), size=size, mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.5


def build_groups(
    images: torch.Tensor,
    disp_u8: torch.Tensor,
    focus: torch.Tensor,
    k: torch.Tensor,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    group_frames: int = DEFAULT_GROUP_FRAMES,
    overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
    mpi_alpha: float = MPI_ALPHA_INFER,
) -> dict[str, torch.Tensor]:
    """Build the stacked tensors the pipeline consumes, replacing BokehCocMPIDatasetWholeVid.

    images: (N,H,W,3) float in [0,1] (ComfyUI IMAGE convention)
    disp_u8: (N,h,w) uint8 disparity, clip-normalised
    focus: (N,) float in [0,1]   -- upstream `zf`
    k: (N,) float                -- upstream blur strength

    Returns dict with (G = number of groups):
      pixel_values (G,F,3,H,W) float32 in [-1,1]
      coc          (G,F,3,H,W) float32 in [-1,1]
      mpi_mask     (G,F,H,W)   float32 region indices in [0, segments-1]
      k            (G,F)       float32
      weight       (G,F)       float32
    """
    size = (height, width)
    n = images.shape[0]
    if disp_u8.shape[0] != n:
        raise ValueError(f"images have {n} frames but disparity has {disp_u8.shape[0]}")
    if focus.shape[0] != n or k.shape[0] != n:
        raise ValueError("focus and k must have one value per frame")

    # (N,H,W,3) [0,1] -> (N,3,H,W) [0,1]  (numpy_to_pt)
    px = images.permute(0, 3, 1, 2).float().contiguous() / 1.0
    if px.max() > 1.5:  # tolerate 0..255 input
        px = px / 255.0

    disp_f = disp_u8.float()
    coc_list, mpi_list = [], []
    for i in range(n):
        d = disp_f[i]
        f_i = float(focus[i].item())
        coc_r = torch.abs(d - f_i * 255.0) / 255.0  # upstream: coc_r
        mpi = gen_mpi_mask(d / 255.0, f_i, alpha=mpi_alpha).float()
        coc_list.append(coc_r.unsqueeze(0).repeat(3, 1, 1))
        mpi_list.append(mpi)
    coc = torch.stack(coc_list)  # (N,3,h,w) in [0,1]
    mpi_mask = torch.stack(mpi_list)  # (N,h,w)
    if mpi_mask.shape[-2:] != size:
        mpi_mask = TF.interpolate(mpi_mask.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)

    px = _resize_norm(px, size)
    coc = _resize_norm(coc, size)

    groups, group_weights = group_frames_with_overlap(n, group_frames, overlap_frames)

    out_px, out_coc, out_mpi, out_k, out_w = [], [], [], [], []
    for gi, (group, weight) in enumerate(zip(groups, group_weights)):
        idx = torch.as_tensor(group, dtype=torch.long)
        g_px, g_coc, g_mpi = px[idx], coc[idx], mpi_mask[idx]
        g_k, g_w = k[idx].float(), torch.as_tensor(weight, dtype=torch.float32)
        if len(group) != group_frames and out_px:
            pad = group_frames - len(group)
            g_px = torch.cat([g_px, g_px.new_zeros(pad, *g_px.shape[1:])])
            g_coc = torch.cat([g_coc, g_coc.new_zeros(pad, *g_coc.shape[1:])])
            g_mpi = torch.cat([g_mpi, g_mpi.new_zeros(pad, *g_mpi.shape[1:])])
            g_k = torch.cat([g_k, g_k.new_zeros(pad)])
            g_w = torch.cat([g_w, g_w.new_zeros(pad)])
        out_px.append(g_px)
        out_coc.append(g_coc)
        out_mpi.append(g_mpi)
        out_k.append(g_k)
        out_w.append(g_w)

    if not out_px:
        raise ValueError("no frame groups produced; is the clip shorter than one frame?")

    return {
        "pixel_values": torch.stack(out_px).contiguous().float(),
        "coc": torch.stack(out_coc).contiguous().float(),
        "mpi_mask": torch.stack(out_mpi).contiguous().float(),
        "k": torch.stack(out_k).contiguous().float(),
        "weight": torch.stack(out_w).contiguous().float(),
        "num_groups": len(out_px),
    }


FOCUS_NEAR_PCT, FOCUS_MID_PCT, FOCUS_FAR_PCT = 90.0, 50.0, 10.0


def resolve_focus_target(target: str, disp_frame_u8: torch.Tensor) -> float:
    """Resolve one keyframe target into a focal plane in [0,1] (1.0 = nearest).

    Accepted forms, where disp is uint8 0..255:
      "0.42"        explicit plane in 0..1
      "180"         explicit plane in 0..255 units (upstream's _zf_ grayscale convention)
      "0.45,0.62"   a POINT: median disparity in a small window around (x, y) in 0..1 coords,
                    i.e. "focus on whatever is under this pixel"
      "near"/"mid"/"far"  percentile of this frame's disparity (90/50/10)
    """
    t = target.strip().lower()
    d = disp_frame_u8.float()
    if t in ("near", "mid", "far"):
        pct = {"near": FOCUS_NEAR_PCT, "mid": FOCUS_MID_PCT, "far": FOCUS_FAR_PCT}[t]
        return float(torch.quantile(d.flatten(), pct / 100.0)) / 255.0
    if "," in t:
        xs, ys = t.split(",")
        x, y = float(xs), float(ys)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"point coords must be in 0..1, got {t!r}")
        h, w = d.shape
        cx, cy = int(round(x * (w - 1))), int(round(y * (h - 1)))
        r = max(1, int(round(0.01 * max(h, w))))
        win = d[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        return float(win.median()) / 255.0
    v = float(t)
    return v / 255.0 if v > 1.0 else v


def parse_focus_keyframes(text: str, disp_u8: torch.Tensor, interp: str = "linear"):
    """'0:0.5,0.4; 60:near; end:far' -> (per-frame focal plane tensor in [0,1], resolved entries).

    Frame is an int index or the word 'end' (last frame). Values before the first key and after the
    last one are held. interp: 'linear' | 'smooth' (cosine ease) | 'hold' (step). A single entry gives
    a FIXED focal plane for the whole clip.
    """
    n = disp_u8.shape[0]
    entries: list[tuple[int, float]] = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        frame_s, sep, target = chunk.partition(":")
        if not sep:
            raise ValueError(f"bad keyframe {chunk!r}, expected frame:target")
        frame_s = frame_s.strip().lower()
        fi = (n - 1) if frame_s == "end" else int(float(frame_s))
        fi = min(max(fi, 0), n - 1)
        entries.append((fi, resolve_focus_target(target, disp_u8[fi])))
    if not entries:
        raise ValueError("no keyframes parsed")
    entries.sort(key=lambda e: e[0])
    vals = torch.tensor([v for _, v in entries], dtype=torch.float32)
    ks = torch.tensor([float(k) for k, _ in entries], dtype=torch.float32)
    idx = torch.arange(n, dtype=torch.float32)
    if len(entries) == 1:
        return vals.repeat(n), entries
    pos = torch.searchsorted(ks, idx, right=True).clamp(1, len(ks) - 1)
    k0, k1 = ks[pos - 1], ks[pos]
    v0, v1 = vals[pos - 1], vals[pos]
    t = ((idx - k0) / (k1 - k0).clamp_min(1e-6)).clamp(0.0, 1.0)
    if interp == "hold":
        t = torch.zeros_like(t)
    elif interp == "smooth":
        t = 0.5 - 0.5 * torch.cos(math.pi * t)
    f = torch.where(idx <= ks[0], vals[0], torch.where(idx >= ks[-1], vals[-1], v0 + (v1 - v0) * t))
    return f.clamp(0.0, 1.0), entries


def focus_curve_image(f: torch.Tensor, entries, width: int = 512, height: int = 160) -> torch.Tensor:
    """Render the f(t) scenario curve as an (H,W,3) image: green line = focal plane over time,
    orange columns = keyframes, red squares = the resolved keyframe values. Top = near, bottom = far.
    """
    n = f.shape[0]
    img = torch.zeros(height, width, 3)
    img[:] = torch.tensor([0.08, 0.08, 0.09])
    for gy in (0.0, 0.5, 1.0):
        img[int((1.0 - gy) * (height - 1)), :] = torch.tensor([0.25, 0.25, 0.28])
    xs = (idx := torch.arange(n, dtype=torch.float32)) / max(n - 1, 1) * (width - 1)
    xs = xs.long()
    ys = ((1.0 - f) * (height - 1)).long().clamp(0, height - 1)
    img[ys, xs] = torch.tensor([0.2, 0.9, 0.4])
    for k, v in entries:
        x = int(k / max(n - 1, 1) * (width - 1))
        img[:, max(0, x - 1):x + 1] = torch.tensor([0.9, 0.6, 0.1])
        y = int((1.0 - v) * (height - 1))
        img[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = torch.tensor([1.0, 0.3, 0.2])
    return img
