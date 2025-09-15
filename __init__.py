from .inpaint_blend import EnhancedImageCompositeMasked

NODE_CLASS_MAPPINGS = {
    "EnhancedImageCompositeMasked": EnhancedImageCompositeMasked,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnhancedImageCompositeMasked": "Enhanced Image Composite Masked",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]