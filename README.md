# ComfyUI-AnyToBokeh

ComfyUI wrapper for **Any-to-Bokeh** (vivoCameraResearch/any-to-bokeh, ICLR 2026) — temporally
coherent, depth-aware **video refocusing**: you set the focal plane and the blur strength, the model
renders the bokeh. One-step video diffusion on a modified SVD backbone, conditioned on a CoC map, a
focal-plane-adapted MPI mask and `k`.

Upstream: https://github.com/vivoCameraResearch/any-to-bokeh · Paper: https://arxiv.org/abs/2505.21593

Depth and masks are **not** reimplemented here — they come from ComfyUI core nodes.