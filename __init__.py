"""ComfyUI-AnyToBokeh_v2: video refocusing (controllable bokeh) inside ComfyUI.

Wraps vivoCameraResearch/any-to-bokeh (ICLR 2026). Depth and masks come from ComfyUI core nodes
(DA3 + SAM3); this pack only builds the tensors the model expects and runs it.

v2 is the slim release: the six nodes the two workflows in ./workflows use (focus from an editable
curve, and focus from a SAM mask), and nothing else.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./web"
