from .inpaint_blend import InpaintBlend

NODE_CLASS_MAPPINGS = {
    "InpaintBlend": InpaintBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintBlend": "Inpaint Blend",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]