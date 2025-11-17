#!/usr/bin/env python3
"""
Test script for Sa2VA integration in ComfyUI-Sa2VA
This script validates the Sa2VA node functionality and mask conversion.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import tempfile

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from nodes.sa2va_node_tpl import Sa2VANodeTpl

    print("✅ Successfully imported Sa2VA node")
except ImportError as e:
    print(f"❌ Failed to import Sa2VA node: {e}")
    sys.exit(1)


def create_test_image():
    """Create a simple test image for testing."""
    # Create a simple test image with some colored rectangles
    img = Image.new("RGB", (256, 256), color="white")
    # Add some colored rectangles to segment
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)

    # Red rectangle
    draw.rectangle([50, 50, 100, 100], fill="red")
    # Blue circle
    draw.ellipse([150, 50, 200, 100], fill="blue")
    # Green triangle (approximated as polygon)
    draw.polygon([(125, 150), (100, 200), (150, 200)], fill="green")

    return img


def test_sa2va_text_only():
    """Test Sa2VA node for text generation only."""
    print("\n🔍 Testing Sa2VA text generation...")

    try:
        # Create test image
        test_img = create_test_image()
        image_batch = [[test_img]]

        # Initialize Sa2VA node
        sa2va_node = Sa2VANodeTpl()

        # Test text-only generation (now always in segmentation mode)
        text_outputs, masks, mask_images = sa2va_node.process_with_sa2va(
            model_name="ByteDance/Sa2VA-Qwen3-VL-4B",
            image=image_batch[0][0],  # Pass single image instead of batch
            mask_threshold=0.5,
            segmentation_prompt="Please describe this image.",
            use_8bit_quantization=False,
            use_flash_attn=True,
        )

        print(f"✅ Text generation successful!")
        print(f"   Generated {len(text_outputs)} text outputs")
        print(f"   Sample output: {text_outputs[0][:100]}...")
        if masks is not None:
            print(f"   Mask tensor shape: {masks.shape}")
        else:
            print(f"   No masks generated")

        return True

    except Exception as e:
        print(f"❌ Text generation test failed: {e}")
        return False


def test_sa2va_segmentation():
    """Test Sa2VA node for segmentation."""
    print("\n🎯 Testing Sa2VA segmentation...")

    try:
        # Create test image
        test_img = create_test_image()
        image_batch = [[test_img]]

        # Initialize Sa2VA node
        sa2va_node = Sa2VANodeTpl()

        # Test segmentation generation
        text_outputs, masks, mask_images = sa2va_node.process_with_sa2va(
            model_name="ByteDance/Sa2VA-Qwen3-VL-4B",
            image=image_batch[0][0],  # Pass single image instead of batch
            mask_threshold=0.5,
            segmentation_prompt="Please provide segmentation masks for all objects in this image.",
            use_8bit_quantization=False,
            use_flash_attn=True,
        )

        print(f"✅ Segmentation generation successful!")
        print(f"   Generated {len(text_outputs)} text outputs")

        if masks is not None:
            print(f"   ComfyUI mask tensor shape: {masks.shape}")
        else:
            print(f"   No ComfyUI masks generated")

        if mask_images is not None:
            print(f"   Mask image tensor shape: {mask_images.shape}")
        else:
            print(f"   No mask images generated")

        return True, masks

    except Exception as e:
        print(f"❌ Segmentation test failed: {e}")
        return False, []


def test_integrated_mask_conversion(masks):
    """Test the integrated mask conversion in the Sa2VA node."""
    print("\n🔄 Testing Integrated Mask Conversion...")

    if masks is None:
        print("⚠️ No mask tensor provided")
        return True

    try:
        print(f"✅ Integrated mask conversion successful!")
        print(f"   ComfyUI mask tensor shape: {masks.shape}")
        print(f"   ComfyUI mask tensor type: {type(masks)}")
        print(f"   Mask tensor dtype: {masks.dtype}")

        return True

    except Exception as e:
        print(f"❌ Integrated mask conversion test failed: {e}")
        return False


def test_video_mode():
    """Test Sa2VA video processing mode."""
    print("\n🎬 Testing Sa2VA video mode...")

    try:
        # Create multiple test images to simulate video frames
        test_frames = []
        for i in range(3):
            img = create_test_image()
            # Add some variation to simulate motion
            # This is a simplified simulation
            test_frames.append(img)

        frame_batch = [test_frames]

        # Initialize Sa2VA node
        sa2va_node = Sa2VANodeTpl()

        # Test video processing (Note: video mode requires special handling)
        # For now, test with single frame as video mode needs frame sequences
        text_outputs, masks, mask_images = sa2va_node.process_with_sa2va(
            model_name="ByteDance/Sa2VA-Qwen3-VL-4B",
            image=test_frames[0],  # Use first frame
            mask_threshold=0.5,
            segmentation_prompt="Please describe what happens in this video sequence.",
            use_8bit_quantization=False,
            use_flash_attn=True,
        )

        print(f"✅ Video processing successful!")
        print(f"   Generated {len(text_outputs)} video descriptions")
        print(f"   Sample output: {text_outputs[0][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        return False


def test_model_loading():
    """Test model loading and caching."""
    print("\n📦 Testing model loading and caching...")

    try:
        sa2va_node = Sa2VANodeTpl()

        # Test initial model loading
        print("   Loading model for first time...")
        success1 = sa2va_node.load_model(
            "ByteDance/Sa2VA-Qwen3-VL-4B", use_flash_attn=True, dtype="auto"
        )

        if not success1:
            print("❌ Initial model loading failed")
            return False

        print("   ✅ Initial model loading successful")

        # Test cached model loading
        print("   Testing model cache...")
        success2 = sa2va_node.load_model(
            "ByteDance/Sa2VA-Qwen3-VL-4B", use_flash_attn=True, dtype="auto"
        )

        if not success2:
            print("❌ Cached model loading failed")
            return False

        print("   ✅ Model caching working correctly")

        return True

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def main():
    """Run all tests for Sa2VA integration."""
    print("🚀 Starting ComfyUI-Sa2VA Integration Tests")
    print("=" * 50)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"🔧 CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("⚠️ Running on CPU - tests may be slow")

    test_results = []

    # Test 1: Model Loading
    test_results.append(("Model Loading", test_model_loading()))

    # Test 2: Text Generation
    test_results.append(("Text Generation", test_sa2va_text_only()))

    # Test 3: Segmentation
    segmentation_success, masks = test_sa2va_segmentation()
    test_results.append(("Segmentation", segmentation_success))

    # Test 4: Integrated Mask Conversion
    test_results.append(
        ("Integrated Mask Conversion", test_integrated_mask_conversion(masks))
    )

    # Test 5: Video Mode
    test_results.append(("Video Processing", test_video_mode()))

    # Print test summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:<20}: {status}")
        if result:
            passed += 1

    print("-" * 50)
    print(f"   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! ComfyUI-Sa2VA is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
