# Changelog

## [1.1.0] - 2024-01-15

### Changed
- Renamed node class to `EnhancedImageCompositeMasked` for better clarity
- Set Poisson blending as the default mode (was previously "default" mode)
- Fixed code style warnings (E702 - multiple statements on one line)

### Added
- Comprehensive documentation for Google Nano Banana and ByteDance Seedream 4 support
- Model-specific usage recommendations in README

### Fixed
- Color mismatch issues when using ByteDance Seedream 4 for inpainting

## [1.0.0] - 2024-01-14

### Initial Release
- Poisson blending implementation using PyTorch Jacobi iterative solver
- Default (alpha) blending mode
- Support for image-space inpainting workflows
- Batch processing support
- GPU acceleration via PyTorch CUDA