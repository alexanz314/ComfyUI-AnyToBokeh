"""Compatibility shim: run the upstream Any-to-Bokeh code on modern diffusers.

Upstream pins diffusers==0.25.0 / transformers==4.32.1. ComfyUI ships diffusers 0.39.0 and
transformers 5.13.1, where several modules moved and a few legacy mixins were removed:

    diffusers.models.transformer_2d   -> diffusers.models.transformers.transformer_2d
    diffusers.models.unet_3d_blocks   -> diffusers.models.unets.unet_3d_blocks
    diffusers.models.unet_2d_blocks   -> diffusers.models.unets.unet_2d_blocks
    diffusers.models.lora             -> removed (LoRACompatible* wrappers)
    diffusers.loaders.FromOriginalVAEMixin -> removed

Rather than patching five vendored files (which would make every upstream update a merge
conflict), this module registers aliases/stubs in sys.modules BEFORE they are imported.
`install()` is idempotent and is a no-op on diffusers 0.25, where everything already resolves.

Verified available in diffusers 0.39.0 (probed, not assumed): AutoencoderKLTemporalDecoder,
UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline, apply_freeu, USE_PEFT_BACKEND,
randn_tensor, maybe_allow_in_graph, VaeImageProcessor, EulerDiscreteScheduler,
transformers.CLIPImageProcessor / CLIPVisionModelWithProjection.
"""

from __future__ import annotations

import importlib
import sys
import types

# legacy module name -> candidate modern locations, tried in order
_ALIASES = {
    "diffusers.models.transformer_2d": ["diffusers.models.transformers.transformer_2d"],
    "diffusers.models.unet_3d_blocks": [
        "diffusers.models.unets.unet_3d_blocks",
        "diffusers.models.blocks.unet_3d_blocks",
    ],
    "diffusers.models.unet_2d_blocks": [
        "diffusers.models.unets.unet_2d_blocks",
        "diffusers.models.blocks.unet_2d_blocks",
    ],
    "diffusers.models.resnet": ["diffusers.models.resnets"],
    "diffusers.models.resnets": ["diffusers.models.resnet"],
}


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def install(verbose: bool = False) -> list[str]:
    """Register aliases/stubs. Returns a human-readable list of what was changed."""
    import torch.nn as nn

    notes: list[str] = []

    for legacy, candidates in _ALIASES.items():
        if legacy in sys.modules or _module_available(legacy):
            continue
        for cand in candidates:
            if _module_available(cand):
                sys.modules[legacy] = sys.modules[cand]
                notes.append(f"alias {legacy} -> {cand}")
                break

    # diffusers.models.lora held LoRA-compatible wrappers around nn.Linear/nn.Conv2d. They were
    # removed; the wrappers were transparent when no LoRA was attached, so the plain layers are a
    # faithful stand-in for inference.
    if not _module_available("diffusers.models.lora"):
        _stub(
            "diffusers.models.lora",
            LoRACompatibleLinear=nn.Linear,
            LoRACompatibleConv=nn.Conv2d,
            LoRACompatibleAttention=nn.Module,
            LoRACompatibleModelMixin=object,
        )
        notes.append("stub diffusers.models.lora (LoRACompatibleLinear -> nn.Linear)")

    # Mixins that only added single-file (.ckpt/.safetensors) loading, which we never use:
    # checkpoints are loaded through diffusers' own from_pretrained on a diffusers directory.
    loaders = importlib.import_module("diffusers.loaders")
    for mixin in ("FromOriginalVAEMixin", "FromOriginalModelMixin", "UNet2DConditionLoadersMixin"):
        if not hasattr(loaders, mixin):
            setattr(loaders, mixin, object)
            notes.append(f"stub diffusers.loaders.{mixin} -> object")

    resnet = sys.modules.get("diffusers.models.resnet") or (
        sys.modules.get("diffusers.models.resnets") if _module_available("diffusers.models.resnets") else None
    )
    if resnet is not None and not hasattr(resnet, "AlphaBlender"):
        class AlphaBlender(nn.Module):
            """Fallback for the removed helper: convex blend of the spatial and temporal branches."""

            def __init__(self, num_features: int, neg_ctrl: float = 0.0, pos_ctrl: float = 1.0):
                super().__init__()
                import torch
                self.mix_factor = nn.Parameter(torch.zeros(1))
                self.num_layers = 1

            def forward(self, spatial, temporal, scale=None):
                a = self.mix_factor.sigmoid()
                return (1.0 - a) * temporal + a * spatial

        resnet.AlphaBlender = AlphaBlender
        notes.append("stub diffusers.models.resnet.AlphaBlender")

    if verbose and notes:
        for n in notes:
            print(f"[a2b-compat] {n}")
    return notes
