# Sa2VA Model Configuration
# Configuration settings for Sa2VA models in ComfyUI-Sa2VA

"""
Configuration file for Sa2VA (Segment Anything 2 Video Assistant) models.
This file contains model specifications, default settings, and optimization parameters
for the ComfyUI-Sa2VA project.
"""

# Available Sa2VA Models
SA2VA_MODELS = {
    "ByteDance/Sa2VA-Qwen3-VL-4B": {
        "name": "Sa2VA-Qwen3-VL-4B",
        "description": "4B parameter model based on Qwen3-VL",
        "recommended_vram": "8GB",
        "performance": "fast",
        "quality": "high",
        "native_dtype": "float32",
        "default_dtype": "auto",
        "supports_flash_attn": True,
        "max_resolution": 1024,
        "video_frames_limit": 8,
        "precision_note": "Native float32 with bfloat16 text components. Auto-conversion supported.",
    },
    "ByteDance/Sa2VA-Qwen2_5-VL-7B": {
        "name": "Sa2VA-Qwen2.5-VL-7B",
        "description": "7B parameter model based on Qwen2.5-VL",
        "recommended_vram": "14GB",
        "performance": "medium",
        "quality": "very_high",
        "native_dtype": "float32",
        "default_dtype": "auto",
        "supports_flash_attn": True,
        "max_resolution": 1024,
        "video_frames_limit": 10,
        "precision_note": "Native float32 with bfloat16 text components. Auto-conversion supported.",
    },
    "ByteDance/Sa2VA-InternVL3-8B": {
        "name": "Sa2VA-InternVL3-8B",
        "description": "8B parameter model based on InternVL3",
        "recommended_vram": "16GB",
        "performance": "slow",
        "quality": "excellent",
        "native_dtype": "float32",
        "default_dtype": "auto",
        "supports_flash_attn": True,
        "max_resolution": 1024,
        "video_frames_limit": 12,
        "precision_note": "Native float32 with bfloat16 text components. Auto-conversion supported.",
    },
    "ByteDance/Sa2VA-InternVL3-14B": {
        "name": "Sa2VA-InternVL3-14B",
        "description": "14B parameter model based on InternVL3",
        "recommended_vram": "24GB",
        "performance": "very_slow",
        "quality": "state_of_art",
        "native_dtype": "float32",
        "default_dtype": "auto",
        "supports_flash_attn": True,
        "max_resolution": 1024,
        "video_frames_limit": 15,
        "precision_note": "Native float32 with bfloat16 text components. Auto-conversion supported.",
    },
    "ByteDance/Sa2VA-Qwen2_5-VL-3B": {
        "name": "Sa2VA-Qwen2.5-VL-3B",
        "description": "3B parameter model based on Qwen2.5-VL",
        "recommended_vram": "6GB",
        "performance": "very_fast",
        "quality": "good",
        "native_dtype": "float32",
        "default_dtype": "auto",
        "supports_flash_attn": True,
        "max_resolution": 1024,
        "video_frames_limit": 6,
        "precision_note": "Native float32 with bfloat16 text components. Auto-conversion supported.",
    },
    "ByteDance/Sa2VA-InternVL3-2B": {
        "name": "Sa2VA-InternVL3-2B",
        "description": "2B parameter model based on InternVL3",
        "recommended_vram": "4GB",
        "performance": "very_fast",
        "quality": "good",
        "native_dtype": "float32",
        "default_dtype": "auto",
        "supports_flash_attn": True,
        "max_resolution": 1024,
        "video_frames_limit": 5,
        "precision_note": "Native float32 with bfloat16 text components. Auto-conversion supported.",
    },
}

# Default Configuration
DEFAULT_CONFIG = {
    "model_name": "ByteDance/Sa2VA-Qwen3-VL-4B",
    "dtype": "auto",
    "use_flash_attn": True,
    "max_frames": 5,
    "segmentation_mode": False,
    "video_mode": False,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}

# Prompt Templates
PROMPT_TEMPLATES = {
    "describe": {
        "text": "Please describe the image in detail.",
        "segmentation": "Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
    },
    "analyze": {
        "text": "Please analyze this image and identify all objects, people, and activities present.",
        "segmentation": "Please analyze this image and provide segmentation masks for all identifiable objects and regions.",
    },
    "segment_objects": {
        "text": "What objects can you see in this image?",
        "segmentation": "Please segment all distinct objects in this image.",
    },
    "segment_people": {
        "text": "Describe any people in this image.",
        "segmentation": "Please segment all people in this image.",
    },
    "segment_specific": {
        "text": "Describe the {object} in this image.",
        "segmentation": "Please segment the {object} in this image.",
    },
    "video_describe": {
        "text": "Please describe what happens in this video sequence.",
        "segmentation": "Please describe this video and provide segmentation masks for key objects across frames.",
    },
    "video_action": {
        "text": "What actions or movements do you see in this video?",
        "segmentation": "Please identify and segment objects involved in actions throughout this video.",
    },
}

# Performance Optimization Settings
OPTIMIZATION_PRESETS = {
    "speed": {
        "dtype": "float16",  # Convert to float16 for speed
        "use_flash_attn": True,
        "batch_size": 1,
        "max_frames": 3,
        "max_resolution": 512,
        "note": "Converts from native float32 to float16 for faster inference",
    },
    "balanced": {
        "dtype": "auto",  # Use native precision
        "use_flash_attn": True,
        "batch_size": 1,
        "max_frames": 5,
        "max_resolution": 1024,
        "note": "Uses model's native precision (float32/bfloat16 mix)",
    },
    "quality": {
        "dtype": "auto",  # Use native precision
        "use_flash_attn": True,
        "batch_size": 1,
        "max_frames": 8,
        "max_resolution": 1024,
        "note": "Uses model's native precision for best quality",
    },
    "memory_efficient": {
        "dtype": "float16",  # Convert to float16 for memory savings
        "use_flash_attn": False,
        "batch_size": 1,
        "max_frames": 3,
        "max_resolution": 512,
        "note": "Converts to float16 and disables flash attention for memory savings",
    },
}

# Hardware-specific configurations
HARDWARE_CONFIGS = {
    "rtx_4090": {
        "recommended_models": [
            "ByteDance/Sa2VA-Qwen3-VL-4B",
            "ByteDance/Sa2VA-Qwen2_5-VL-7B",
        ],
        "optimal_dtype": "auto",
        "max_batch_size": 2,
        "use_flash_attn": True,
    },
    "rtx_3090": {
        "recommended_models": ["ByteDance/Sa2VA-Qwen3-VL-4B"],
        "optimal_dtype": "auto",
        "max_batch_size": 1,
        "use_flash_attn": True,
    },
    "rtx_3080": {
        "recommended_models": [
            "ByteDance/Sa2VA-InternVL3-2B",
            "ByteDance/Sa2VA-Qwen2_5-VL-3B",
        ],
        "optimal_dtype": "float16",
        "max_batch_size": 1,
        "use_flash_attn": True,
    },
    "cpu_only": {
        "recommended_models": ["ByteDance/Sa2VA-InternVL3-2B"],
        "optimal_dtype": "float32",
        "max_batch_size": 1,
        "use_flash_attn": False,
    },
}

# Segmentation Post-processing Settings
MASK_PROCESSING = {
    "default_threshold": 0.5,
    "default_normalize": True,
    "supported_formats": ["comfyui_mask", "pil_image", "numpy_array"],
    "default_output_format": "both",
}

# Error Messages and Troubleshooting
ERROR_MESSAGES = {
    "cuda_oom": "CUDA out of memory. Try reducing batch size, using a smaller model, or enabling memory-efficient mode.",
    "model_not_found": "Model not found. Please check the model name and ensure you have internet connectivity for initial download.",
    "flash_attn_error": "Flash attention not available. Install flash-attn package or disable use_flash_attn.",
    "unsupported_image": "Unsupported image format. Please ensure images are in PIL Image, tensor, or numpy array format.",
    "video_too_long": "Video sequence too long. Reduce max_frames or split into smaller segments.",
}


# Utility Functions
def get_model_config(model_name):
    """Get configuration for a specific model."""
    return SA2VA_MODELS.get(model_name, {})


def get_recommended_config_for_vram(vram_gb):
    """Get recommended model and settings based on available VRAM."""
    if vram_gb >= 24:
        return "ByteDance/Sa2VA-InternVL3-14B", OPTIMIZATION_PRESETS["quality"]
    elif vram_gb >= 16:
        return "ByteDance/Sa2VA-InternVL3-8B", OPTIMIZATION_PRESETS["balanced"]
    elif vram_gb >= 14:
        return "ByteDance/Sa2VA-Qwen2_5-VL-7B", OPTIMIZATION_PRESETS["balanced"]
    elif vram_gb >= 8:
        return "ByteDance/Sa2VA-Qwen3-VL-4B", OPTIMIZATION_PRESETS["balanced"]
    elif vram_gb >= 6:
        return "ByteDance/Sa2VA-Qwen2_5-VL-3B", OPTIMIZATION_PRESETS["speed"]
    else:
        return "ByteDance/Sa2VA-InternVL3-2B", OPTIMIZATION_PRESETS["memory_efficient"]


def get_prompt_template(template_name, object_name=None):
    """Get prompt template by name."""
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["describe"])
    if object_name and "{object}" in str(template):
        return {
            "text": template["text"].format(object=object_name),
            "segmentation": template["segmentation"].format(object=object_name),
        }
    return template


def validate_config(config):
    """Validate configuration settings."""
    errors = []

    if config.get("model_name") not in SA2VA_MODELS:
        errors.append(f"Unknown model: {config.get('model_name')}")

    if config.get("dtype") not in ["auto", "float16", "bfloat16", "float32"]:
        errors.append(f"Invalid dtype: {config.get('dtype')}")

    if config.get("max_frames", 0) < 1 or config.get("max_frames", 0) > 20:
        errors.append("max_frames must be between 1 and 20")

    return errors


# Version and compatibility info
SA2VA_CONFIG_VERSION = "1.0.0"
COMPATIBLE_TRANSFORMERS_VERSION = ">=4.57.0"
COMPATIBLE_TORCH_VERSION = ">=2.0.0"
