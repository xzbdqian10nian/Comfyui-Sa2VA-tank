# ComfyUI-Sa2VA Troubleshooting Guide

This document helps resolve common issues when using ComfyUI-Sa2VA.

## Common Errors and Solutions

### 1. "No module named 'transformers.models.qwen3_vl'"

**Error Message:**
```
❌ Error loading Sa2VA model ByteDance/Sa2VA-Qwen3-VL-4B: No module named 'transformers.models.qwen3_vl'
```

**Cause:** Your transformers library version is too old and doesn't support Qwen3-VL models.

**Solution:**
```bash
# Upgrade transformers to the latest version
pip install transformers>=4.57.0 --upgrade

# Or if using conda:
conda install transformers>=4.57.0
```

**Note:** Sa2VA models require transformers >= 4.57.0. Earlier versions don't include Qwen3-VL support.

### 2. "'NoneType' object is not subscriptable"

**Error Message:**
```
TypeError: 'NoneType' object is not subscriptable
```

**Cause:** The Sa2VA node failed to load the model, returning None instead of valid outputs.

**Solution:**
1. Check the console for earlier error messages about model loading
2. Ensure transformers version is correct (see error #1)
3. Check internet connectivity for initial model download
4. Verify you have sufficient VRAM (see VRAM requirements below)

### 3. CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Use float16 precision:**
   - Change `dtype` from "auto" to "float16" in Sa2VA node
   
2. **Use smaller model:**
   - Switch to `ByteDance/Sa2VA-InternVL3-2B` or `ByteDance/Sa2VA-Qwen2_5-VL-3B`
   
3. **Reduce batch size:**
   - Process fewer images at once
   
4. **Clear VRAM:**
   ```bash
   # In Python console
   import torch
   torch.cuda.empty_cache()
   ```

### 4. Model Download Issues

**Error Message:**
```
OSError: Can't load the model. Check your internet connection.
```

**Solutions:**
1. **Check internet connection**
2. **Clear HuggingFace cache:**
   ```bash
   # Delete cache directory
   rm -rf ~/.cache/huggingface/
   ```
3. **Manual download:**
   ```bash
   # Pre-download the model
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('ByteDance/Sa2VA-Qwen3-VL-4B', trust_remote_code=True)"
   ```
4. **Use ComfyUI cancel and resume:**
   - Large downloads show live progress and respect ComfyUI’s cancel button.
   - If you cancel midway, simply re-run the node later; downloads will resume from where they left off.
   - Ensure you have sufficient free disk space in the local cache directory:
     - Default for this node: `ComfyUI/custom_nodes/ComfyUI-Sa2VA/.cache/huggingface/hub/`

**Note:** Ensure you have transformers >= 4.57.0 before attempting model download.

### 5. "trust_remote_code" Warning

**Warning Message:**
```
A new version of the following files was downloaded from https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B:
- configuration_sa2va_chat.py
. Make sure to double-check they do not contain any added malicious code.
```

**Explanation:** This is a security warning from HuggingFace. Sa2VA models require custom code to run.

**Solution:** This is normal and expected. The warning appears because Sa2VA uses custom model code that's not in the standard transformers library.

### 6. Flash Attention Errors

**Error Message:**
```
ImportError: Flash attention is not available
```

**Solution:**
1. **Disable flash attention:**
   - Set `use_flash_attn` to `False` in Sa2VA node
   
2. **Install flash attention (optional):**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

## Hardware Requirements

### VRAM Requirements by Model and Precision

| Model | Auto (float32) | float16 | Minimum GPU |
|-------|---------------|---------|-------------|
| Sa2VA-InternVL3-2B | ~8GB | ~4GB | RTX 3070 |
| Sa2VA-Qwen2_5-VL-3B | ~12GB | ~6GB | RTX 3080 |
| Sa2VA-Qwen3-VL-4B | ~20GB | ~8GB | RTX 3090 |
| Sa2VA-Qwen2_5-VL-7B | ~28GB | ~14GB | RTX 4090 |
| Sa2VA-InternVL3-8B | ~32GB | ~16GB | RTX 4090 |

### CPU Requirements
- **Minimum:** 16GB RAM for small models
- **Recommended:** 32GB+ RAM for larger models
- **CPU Processing:** Very slow, not recommended for production use

## Installation Verification

### Test Your Installation
Run this test script to verify everything works:

```python
# Save as test_installation.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    from transformers import __version__
    print(f"Transformers version: {__version__}")
    
    # Check version requirement
    version_parts = __version__.split(".")
    major, minor = int(version_parts[0]), int(version_parts[1])
    if major < 4 or (major == 4 and minor < 57):
        print(f"❌ Transformers {__version__} is too old. Sa2VA requires >= 4.57.0")
    else:
        # Test if transformers supports Qwen3-VL
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("ByteDance/Sa2VA-Qwen3-VL-4B", trust_remote_code=True)
        print("✅ Sa2VA model configuration loaded successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

### Quick Model Test
```python
# Minimal Sa2VA test
from transformers import AutoProcessor, AutoModel
import torch

model_name = "ByteDance/Sa2VA-Qwen3-VL-4B"

try:
    # Load with minimum settings
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Sa2VA model loaded successfully")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
```

## Performance Optimization

### For Low VRAM Systems
1. Use `dtype="float16"`
2. Use smaller models (2B or 3B parameter variants)
3. Process images one at a time
4. Enable `low_cpu_mem_usage=True` (enabled by default)

### For Fast Processing
1. Use `dtype="auto"` for best quality
2. Enable `use_flash_attn=True`
3. Use larger batch sizes if VRAM allows
4. Use GPU with high memory bandwidth

### For CPU-Only Systems
1. Use the smallest model (2B parameters)
2. Use `dtype="float32"`
3. Disable flash attention
4. Process very small images
5. Be patient - CPU inference is slow

## Common Workflow Issues

### No Masks Generated
**Cause:** Segmentation mode not enabled or model didn't detect objects.

**Solution:**
1. Enable `segmentation_mode=True`
2. Use appropriate segmentation prompts
3. Ensure input images contain clear objects
4. Try different models

### Poor Segmentation Quality
**Solutions:**
1. Use more specific prompts: "Please segment the person" instead of "Describe the image"
2. Use higher resolution images
3. Try different Sa2VA model variants
4. Adjust threshold in Mask Converter

### Slow Processing
**Solutions:**
1. Use float16 precision
2. Reduce max_frames for video
3. Use smaller models
4. Enable flash attention
5. Use GPU instead of CPU

## Getting Help

### Before Asking for Help
1. Check this troubleshooting guide
2. Run the installation verification script
3. Check ComfyUI console for error messages
4. Note your system specifications (GPU, VRAM, OS)

### Information to Include
When reporting issues, please include:
- ComfyUI version
- Sa2VA node version
- GPU model and VRAM
- Operating system
- Python version
- Full error message from console
- Workflow JSON (if applicable)

### Where to Get Help
- [GitHub Issues](https://github.com/adambarbato/ComfyUI-Sa2VA/issues)
- [GitHub Discussions](https://github.com/adambarbato/ComfyUI-Sa2VA/discussions)
- ComfyUI Discord community

## Version Compatibility

### Supported Versions
- **Python:** 3.8 - 3.11
- **PyTorch:** >= 2.0.0
- **Transformers:** >= 4.57.0
- **ComfyUI:** Latest stable version

### Known Issues
- **Windows:** May require Visual Studio Build Tools for some dependencies
- **macOS:** Limited GPU support, CPU processing only
- **Linux:** Best compatibility, recommended for production

## Advanced Debugging

### Enable Verbose Logging
```python
# In Sa2VA config.py, set:
be_quiet = False
```

### Check Model Files
```python
# Verify model download
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_dir = f"{cache_dir}/models--ByteDance--Sa2VA-Qwen3-VL-4B"
print(f"Model cached: {os.path.exists(model_dir)}")
```

### Memory Monitoring
```python
# Monitor GPU memory usage
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")

# Call before and after model operations
print_gpu_memory()
```

## Last Resort Solutions

### Complete Reinstall
```bash
# Remove existing installation
rm -rf ComfyUI/custom_nodes/ComfyUI-Sa2VA

# Clear Python cache
pip cache purge

# Fresh install
cd ComfyUI/custom_nodes
git clone https://github.com/adambarbato/ComfyUI-Sa2VA.git
cd ComfyUI-Sa2VA
pip install -r requirements.txt --force-reinstall
```

### Factory Reset ComfyUI
If all else fails, start with a fresh ComfyUI installation and only install ComfyUI-Sa2VA.

---

**Note:** This troubleshooting guide is updated regularly. Check for the latest version if you're having issues not covered here.