from .image_composite_enhanced import ImageCompositeMaskedEnhanced

NODE_CLASS_MAPPINGS = {
    "ImageCompositeMaskedEnhanced": ImageCompositeMaskedEnhanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCompositeMaskedEnhanced": "Image Composite Masked (Enhanced)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]