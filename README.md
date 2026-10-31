# ComfyUI-AnyToBokeh

Unofficial nodes for ComfyUI based on [ICLR2026] Any-to-Bokeh: Arbitrary-Subject Video Refocusing with Video Diffusion Model.

Upstream: https://github.com/vivoCameraResearch/any-to-bokeh · Paper: https://arxiv.org/abs/2505.21593

Depth and masks are **not** reimplemented here — they come from ComfyUI core nodes.

## Example

https://github.com/user-attachments/assets/8780c2cb-989a-4838-afe4-35040e0b255a

Focus changes are configured via the `Any-to-Bokeh: Focus Control (curve / advanced text)` node. The format is "frame_number:relative_depth;" - with each keyframe entry separated by a semicolon.

<img width="559" height="517" alt="curve" src="https://github.com/user-attachments/assets/aed45dd1-c604-4df8-91cb-c12eea00163c" />

## Installation
1. Go to `ComfyUI/custom_nodes`
2. `git clone https://github.com/alexanz314/ComfyUI-AnyToBokeh`
3. Restart ComfyUI

## Required weights

| Path example | Source |
|---|---|
| `models/any2bokeh/unet/` (7.5 GB) | upstream Google Drive, [see their README](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/README.md#-demo-inference) |
| `models/any2bokeh/vae/` (393 MB) | same archive |
| `models/any2bokeh/svd_xt/` | [stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) |
| `models/geometry_estimation/depth_anything_3_mono_large.safetensors` | [Comfy-Org/Depth-Anything-3](https://huggingface.co/Comfy-Org/Depth-Anything-3) |
| (optionally) `models/checkpoints/sam3.1_multiplex_fp16.safetensors` | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) |

## Paths in workflows
Don't forget to change the paths in the node `AnyToBokehLoadModel`! A non-native weights loader is used.
In the `AnyToBokehGenerate` node, specify the path to Python.


I tested the code on Ubuntu using a graphics card with 24 GB of VRAM. It worked.
