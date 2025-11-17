# ComfyUI-Sa2VA - Sa2VA Segmentation Nodes for ComfyUI

from .config import be_quiet  # Import the global config

# Import Sa2VA nodes
from .nodes.sa2va_node_tpl import Sa2VANodeTpl, Sa2VALoaderNode

# Define node class mappings
NODE_CLASS_MAPPINGS = {
    "Sa2VANodeTpl": Sa2VANodeTpl,
    "Sa2VALoaderNode": Sa2VALoaderNode,
}

# Define node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sa2VANodeTpl": "Sa2VA Segmentation",
    "Sa2VALoaderNode": "Sa2VA Loader",
}

# Expose the mappings to ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
