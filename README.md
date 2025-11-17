# ComfyUI Sa2VA

## Overview
A ComfyUI node implementation for [ByteDance's Sa2VA](https://github.com/bytedance/Sa2VA) (Segment Anything 2 Video Assistant) models, enabling advanced multimodal image and video understanding with precise segmentation capabilities.
This repo only implements the image portion of the model.

### What is Sa2VA?
Sa2VA is a state-of-the-art multimodal large language model (MLLM) that combines SAM2 (Segment Anything Model 2) with VLLMs for grounded understanding of images and videos. It achieves comparable performance to SOTA MLLMs like Qwen2.5-VL and InternVL3 on question-answering benchmarks while adding advanced visual prompt understanding and dense object segmentation capabilities.

### Comparisons and Uses
This Sa2VA node can be thought of as a more advanced version of [neverbiasu's ComfyUI-SAM2 node](https://github.com/neverbiasu/ComfyUI-SAM2) that allows for segmentation of objects in an image using natural langauge. Unlike that node which is based on [Grounded SAM/Grounding DINO](https://github.com/IDEA-Research/Grounded-SAM-2), Sa2VA uses a full VLLM trained to output SAM2 segmentation masks, which means it can handle significantly longer and more descriptive text. This allows Sa2VA to be better for uses cases where simple phrases like "woman on right" isn't sufficient to completely disambiguate between objects.

It outperforms Grounding DINO on short prompts:
![](https://raw.githubusercontent.com/adambarbato/ComfyUI-Sa2VA/refs/heads/main/docs/sa2va-grounding-dino.jpg)

And can follow longer instructions quite well, such as describing a character in general, rather than their position or traits in the image itself. This lends itself well to auto-generated or agentic segmentation prompts:
![](https://raw.githubusercontent.com/adambarbato/ComfyUI-Sa2VA/refs/heads/main/docs/long-prompt.jpg)

It can also segment more than one mask at a time, but the prompt needs to be precise:
![](https://raw.githubusercontent.com/adambarbato/ComfyUI-Sa2VA/refs/heads/main/docs/multi-mask.jpg)

## Installation

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/adambarbato/ComfyUI-Sa2VA.git
cd ComfyUI-Sa2VA
python install.py
```

### Quick Install (Advanced Users)
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/adambarbato/ComfyUI-Sa2VA.git
cd ComfyUI-Sa2VA
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- **transformers >= 4.57.0** (Critical!)
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ VRAM recommended for 4B models
- 20GB+ VRAM for full precision

**Important:** Sa2VA models require:
- transformers >= 4.57.0 (for Qwen3-VL support)
- qwen_vl_utils (for model utilities)

Older transformers versions will fail with "No module named 'transformers.models.qwen3_vl'" error.

## Features

### **Core Capabilities**
- **Multimodal Understanding**: Combines text generation with visual understanding
- **Dense Segmentation**: Pixel-perfect object segmentation masks
- **Video Processing**: Temporal understanding across video sequences
- **Visual Prompts**: Understanding of spatial relationships and object references
- **Integrated Mask Conversion**: Built-in conversion to ComfyUI mask and image formats
- **Cancellable, Real-time Downloads**: Large model downloads show progress and respect ComfyUI's cancel button

### **Supported Models**
- `ByteDance/Sa2VA-Qwen3-VL-4B` (recommended - 4B parameters)
- `ByteDance/Sa2VA-Qwen2_5-VL-7B` (7B parameters)
- `ByteDance/Sa2VA-InternVL3-8B` (8B parameters)
- `ByteDance/Sa2VA-InternVL3-14B` (14B parameters)
- `ByteDance/Sa2VA-Qwen2_5-VL-3B` (3B parameters)
- `ByteDance/Sa2VA-InternVL3-2B` (2B parameters)

## Node

### **Sa2VA Segmentation**
A single, comprehensive node that provides:
- **Text Generation**: Detailed image descriptions and analysis
- **Object Segmentation**: Precise pixel-level object masks
- **Integrated Mask Conversion**: Automatic conversion to ComfyUI MASK and IMAGE formats
- **Advanced Mask Refinement**: Black/white point control, dilate/erode passes, detail methods
- **Memory Management**: Built-in VRAM optimization and model lifecycle management
- **Dedicated Loader Node**: Load Sa2VA models once and re-use across multiple branches
- **Multiple Output Formats**: Text, ComfyUI masks, and visualizable mask images

## Quick Start

### Basic Image Description
1. Add **Load Image** node and load your image
2. Add **Sa2VA Segmentation** node
3. Connect Load Image → Sa2VA node
4. Adjust `model_name` and `mask_threshold` as needed
5. Set `segmentation_prompt`: "Please describe the image in detail."
6. Execute to get text descriptions

### Image Segmentation
1. Load image using **Load Image** node
2. Add **Sa2VA Segmentation** node
3. Connect Load Image → Sa2VA node
4. Adjust `model_name` and `mask_threshold` as needed
5. Set `segmentation_prompt`: "Please provide segmentation masks for all objects."
6. Connect the `masks` output to mask-compatible nodes or `mask_images` to **Preview Image**
7. The node automatically provides both MASK tensors and visualizable IMAGE tensors

### Mask Refinement Parameters
- **mask_threshold**: Binary threshold after refinement (default 0.30). Combine with `apply_mask_threshold`.
- **apply_mask_threshold**: Enable/disable hard thresholding (defaults to enabled, matching GroundingDINO-style workflow).
- **Black/White Points**: Remap mask luminance (e.g., 0.05 / 0.95) to tighten or loosen selections.
- **Detail Method**: Choose from `VITMatte` or `FastMatte` refinements (closing / smoothing passes).
- **Detail Erode / Dilate**: Integer iterations (0-10) applied after detail method for edge cleanup.
- **Process Details**: Toggle the refinement stack entirely if you prefer raw masks.

### Sa2VA Loader Node
- **Sa2VA Loader** (new node) loads a model once and outputs a `SA2VA_MODEL` handle.
- Connect that handle to multiple Sa2VA Segmentation nodes to share the same weights across threads/batches.
- Great for multi-branch workflows or ComfyUI's queue-based batch processing—no repeated downloads or initializations.
- Loader node lives under the same **Sa2VA** category.

### Video Processing (Issue #2)
**Note:** The current implementation focuses on image processing. For video processing:

1. **Frame-by-Frame Processing:**
   - Use ComfyUI's video loading nodes to extract frames
   - Process each frame individually with Sa2VA node
   - Combine results using batch processing nodes

2. **Video Workflow Example:**
   ```
   Load Video → Extract Frames → Batch Process → Sa2VA Segmentation → Combine Masks
   ```

3. **Future Support:**
   - The codebase includes `process_video_frames()` method for future video support
   - Full video temporal understanding requires model updates from ByteDance
   - Current Sa2VA models support single-frame analysis only

4. **Workaround for Video:**
   - Extract key frames from video
   - Process frames as batch using ComfyUI batch nodes
   - Use temporal smoothing nodes to interpolate between frames

## Model Location (Issue #1)

Models are automatically downloaded when first used. The default storage location is:
- **ComfyUI Models Directory:** `ComfyUI/models/sa2va/模型名/` (recommended)
- **Fallback:** `ComfyUI-Sa2VA/.cache/huggingface/hub/` (if ComfyUI structure not detected)

**Important:** Models are stored with **clear folder structure** (not blob cache), making them easy to find and manage:
- Each model has its own folder with readable name (e.g., `Sa2VA-Qwen3-VL-4B`)
- All model files are directly visible (no hash filenames)
- You can manually download models and place them in the corresponding folder

**Directory Structure:**
```
ComfyUI/
  models/
    sa2va/
      Sa2VA-Qwen3-VL-4B/          # Clear model folder
        config.json
        model files...
      Sa2VA-Qwen2_5-VL-7B/        # Another model
        config.json
        model files...
```

To check model location:
1. Look in the console output when loading a model - it shows the directory
2. Each model is typically 4-20GB in size
3. Models are located at: `ComfyUI/models/sa2va/模型名/`

**Manual Model Installation:**
You can manually download models from HuggingFace and place them in `ComfyUI/models/sa2va/模型名/` folder. The node will automatically detect and use them.

## Model Loader & Multi-Threading

- Use the **Sa2VA Loader** node to preload models when running multi-threaded or multi-branch workflows.
- The loader keeps the model resident on the selected device; downstream segmentation nodes just reference it.
- To free VRAM per run, toggle `clean_vram_after` on the segmentation node. Leave it off when you share the loader output.

## Model Precision

Sa2VA models use **bfloat16 precision** by default with the option to quantize to 8 bits using bits-and-bytes.


## Troubleshooting

### Common Issues

**"No module named 'transformers.models.qwen3_vl'"**
```bash
pip install transformers>=4.57.0 --upgrade
```
This is the most common issue - your transformers version is too old.

**"No module named 'qwen_vl_utils'"**
```bash
pip install qwen_vl_utils
```
This dependency is required for Sa2VA model utilities.

**"'NoneType' object is not subscriptable"**
- Model loading failed (check console for errors)
- Usually caused by outdated transformers version
- Verify internet connection for model download

**CUDA Out of Memory**
- Use 8 bit quantization
- Use smaller model variant (2B or 3B parameters)
- Reduce batch size

**Model Loading Errors**
- Check internet connection for initial download
- Ensure sufficient disk space (20GB+ per model)
- Verify CUDA compatibility
- Try: `torch.cuda.empty_cache()` to clear VRAM

**Poor Segmentation Quality**
- Use more specific prompts: "Please segment the person"
- Try different model variants
- Adjust threshold in Mask Converter
- Use higher resolution images

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Advanced Configuration

### Custom Prompts
```python
# Object-specific segmentation
"Please segment the person in the image"
"Identify and segment all vehicles"

# Multi-object segmentation
"Create separate masks for all distinct objects"
"Segment foreground and background separately"
```

## Contributing

Contributions welcome! Areas for improvement:
- Performance optimizations
- Video processing
- Better mask post-processing

## License

MIT

## Testing Installation

Run the test script to verify everything works:
```bash
cd ComfyUI-Sa2VA
python test_sa2va.py
```

## Links

- [Sa2VA Paper](https://arxiv.org/abs/2501.04001)
- [Sa2VA Models on HuggingFace](https://huggingface.co/ByteDance)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Based on code from [ComfyUI-Transformers-Pipeline](https://github.com/mediocreatmybest/ComfyUI-Transformers-Pipeline)
- [Issues & Support](https://github.com/adambarbato/ComfyUI-Sa2VA/issues)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
