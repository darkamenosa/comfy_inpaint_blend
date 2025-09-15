# ComfyUI Image Composite Enhanced

An enhanced version of the ImageCompositeMasked node for ComfyUI with **Poisson blending** support for seamless image compositing. This implementation is specifically designed to solve the common problem in image inpainting where the generated content has good texture but mismatched colors compared to the surrounding area.

## Why Poisson Blending?

Traditional alpha blending often creates visible seams when compositing images with different lighting or color conditions. **Poisson blending solves this by**:
- **Preserving textures and details** from the source image (inpainted region)
- **Automatically matching colors** to the destination image (original image)
- **Creating seamless transitions** at boundaries without visible edges
- **Ideal for inpainting workflows** where AI-generated content has correct structure but wrong colors

## Features

- **Default Mode**: Traditional alpha/mask compositing
- **Poisson Mode**: Advanced gradient-domain blending
  - Preserves source gradients (textures, edges, details)
  - Adjusts colors to match destination seamlessly
  - Eliminates visible seams at mask boundaries
  - Perfect for fixing color mismatches in inpainting results

## Installation

1. Navigate to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. Clone or copy this repository:
   ```bash
   git clone <repository-url> comfy_compose
   ```

3. Install dependencies:
   ```bash
   pip install -r comfy_compose/requirements.txt
   ```

4. Restart ComfyUI

## Usage

The node appears as **"Image Composite Masked (Enhanced)"** in the image category.

### Typical Inpainting Workflow

1. Generate inpainted content with your preferred model
2. Use this node to composite the inpainted region back to the original image
3. Set `blend_mode` to `"poisson"` for seamless color matching
4. The result will preserve the AI-generated textures while matching surrounding colors

### Inputs

- **destination**: The original/background image
- **source**: The inpainted/generated image to composite
- **x**: X position for placement (0 to MAX_RESOLUTION)
- **y**: Y position for placement (0 to MAX_RESOLUTION)
- **resize_source**: Whether to resize source to fit destination
- **blend_mode**:
  - `"default"`: Standard alpha compositing (may show seams)
  - `"poisson"`: Gradient-domain blending (seamless, color-matched)
- **mask** (optional): Inpainting mask defining the blend region

### Output

- **IMAGE**: The seamlessly composited result

## Poisson Blending Algorithm

The Poisson mode implements the algorithm from "Poisson Image Editing" (Pérez et al., 2003). This technique is particularly effective for inpainting workflows where the AI generates good textures but incorrect colors.

### How It Works

The algorithm solves the Poisson equation **∇²f = ∇²g** where:
- **f**: The final blended image we're solving for
- **g**: The source image (inpainted content) whose gradients we preserve
- **Boundary conditions**: Colors from the destination image at mask edges

### Implementation Details

Currently using a **pure PyTorch Jacobi iterative solver**:
- Preserves source image gradients (textures, edges, patterns)
- Enforces destination colors at boundaries
- Iteratively propagates color corrections inward
- Converges to a seamless blend that matches surrounding colors

### Why This Solves Inpainting Color Mismatch

1. **Gradient Preservation**: Keeps all the texture details from the AI-generated content
2. **Color Harmonization**: Automatically adjusts the overall color/lighting to match surroundings
3. **Seamless Boundaries**: No visible seams or color jumps at mask edges
4. **Structure Retention**: Maintains the structural integrity of the inpainted content

## Technical Details

### Performance Considerations

- **Iterative Solver**: Uses up to 400 iterations with convergence tolerance of 1e-4
- **GPU Acceleration**: Fully leverages PyTorch CUDA for fast computation
- **Memory Efficient**: Operates directly on image tensors without sparse matrices
- **Batch Processing**: Supports batched operations for video/animation workflows

### Algorithm Parameters

The Poisson solver uses:
- **Jacobi iterations**: Stable convergence for the discrete Poisson equation
- **4-neighbor Laplacian**: Standard discrete approximation of gradient operator
- **Replicate padding**: Handles image boundaries naturally
- **Early stopping**: Exits when convergence tolerance is reached

## Use Cases

### Perfect for:
- **Inpainting workflows**: When AI generates correct content but wrong colors
- **Object insertion**: Adding objects that need to match scene lighting
- **Photo restoration**: Blending restored patches seamlessly
- **Style transfer**: Combining content with different color schemes
- **Texture synthesis**: Merging synthesized textures without visible seams

### Example Workflow for Inpainting:
1. Use any inpainting model (SD, DALL-E, etc.) to generate content
2. Connect the original image to `destination`
3. Connect the inpainted result to `source`
4. Provide the inpainting mask
5. Set `blend_mode` to `"poisson"`
6. The output will have the AI's textures with correct colors

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0 (with CUDA support recommended)
- NumPy >= 1.19.0
- OpenCV >= 4.5.0 (optional, for future optimizations)
- ComfyUI (latest version)

## License

MIT