"""
Style Transfer with Text Prompt Task (GPU Optimized)
Based on: StyleTransfer/transfer_with_text.py
Input: content image + text style description + prompt
Output: Styled image with original background preserved

Optimized to minimize CPU usage and maximize GPU utilization.
"""

import torch
from PIL import Image, ImageFilter
import numpy as np
import cv2
import json
import os
import sys

try:
    from rembg import remove, new_session
except ImportError:
    print(" [ERROR] Error: 'rembg' library is missing.")
    print("   Please install it using: pip install rembg")
    print("   For GPU support: pip install onnxruntime-gpu")
    sys.exit(1)

# Global GPU session for rembg (reused across calls)
_REMBG_SESSION = None


def get_rembg_session():
    """Get or create a GPU-accelerated rembg session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        # Try GPU first, fall back to CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            _REMBG_SESSION = new_session("u2net", providers=providers)
            print("[INFO]: rembg session created with GPU support")
        except Exception as e:
            print(f"[WARN]: Could not create GPU rembg session: {e}")
            _REMBG_SESSION = new_session("u2net")
    return _REMBG_SESSION


def resize_smart(image, max_side=1024):
    """Resize image to fit within max_side while maintaining aspect ratio."""
    w, h = image.size
    if w > h:
        new_w = max_side
        new_h = int(h * (max_side / w))
    else:
        new_h = max_side
        new_w = int(w * (max_side / h))
    new_w = max(8, new_w - (new_w % 8))
    new_h = max(8, new_h - (new_h % 8))
    return image.resize((new_w, new_h), resample=Image.LANCZOS)


def get_canny_image(image):
    """Extract Canny edges from image for ControlNet (GPU accelerated if available)."""
    arr = np.array(image.convert("RGB"))

    # Try GPU-accelerated Canny if OpenCV CUDA is available
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(arr)
            # Convert to grayscale on GPU
            gray_gpu = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_RGB2GRAY)
            # Canny on GPU
            canny_detector = cv2.cuda.createCannyEdgeDetector(100, 200)
            edges_gpu = canny_detector.detect(gray_gpu)
            edges = edges_gpu.download()
            print("[INFO]: Using GPU-accelerated Canny edge detection")
        else:
            raise RuntimeError("No CUDA devices")
    except Exception:
        # Fallback to CPU Canny
        edges = cv2.Canny(arr, 100, 200)

    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def composite_result(original, styled):
    """
    Composite styled object onto original background using rembg (GPU accelerated).
    Keeps the original background and only styles the foreground object.
    """
    print("[INFO]: Segmenting object from original image (using rembg on GPU)...")

    # Resize original to match styled output
    target_size = styled.size
    original_resized = original.resize(target_size, Image.LANCZOS)

    # Create mask from original image using GPU session
    session = get_rembg_session()
    masked_original = remove(original_resized, session=session)

    # Extract alpha channel as mask
    mask = masked_original.split()[-1]

    # Refine mask with feathering
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=2))

    # Composite styled image onto original background
    print("[INFO]: Merging styled object onto original background...")
    final_composite = Image.composite(styled, original_resized, mask_blurred)

    return final_composite


def generate_style_reference(text_prompt, steps, device, dtype):
    """Generate a style reference image from text using SDXL."""
    from diffusers import StableDiffusionXLPipeline

    sdxl_model = "stabilityai/stable-diffusion-xl-base-1.0"

    print(f"[INFO]: Generating style reference for: '{text_prompt}'")

    # Load text-to-image pipeline
    temp_pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if device == "cuda":
        temp_pipe = temp_pipe.to("cuda")
        # GPU memory optimizations without CPU offload
        temp_pipe.enable_vae_slicing()
        temp_pipe.enable_vae_tiling()

    img = temp_pipe(prompt=text_prompt, num_inference_steps=steps).images[0]

    # Cleanup to save VRAM
    del temp_pipe
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return img


def load_controlnet_pipeline(device, dtype, use_cpu_offload=False):
    """
    Load SDXL + ControlNet + IP-Adapter pipeline.

    Args:
        device: 'cuda' or 'cpu'
        dtype: torch dtype
        use_cpu_offload: If True, enables CPU offload (uses more CPU but less VRAM).
                         Set to False to minimize CPU usage.
    """
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    sdxl_model = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model = "diffusers/controlnet-canny-sdxl-1.0"
    ip_adapter = "h94/IP-Adapter"

    print("[INFO]: Loading ControlNet model...")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=dtype,
    )

    print("[INFO]: Loading SDXL+ControlNet pipeline...")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sdxl_model,
        controlnet=controlnet,
        torch_dtype=dtype,
    )

    if device == "cuda":
        pipe = pipe.to("cuda")
        # GPU memory optimizations that DON'T use CPU
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

    print(f"[INFO]: Pipeline loaded on device: {device}")

    # Load IP Adapter
    try:
        print("[INFO]: Loading IP-Adapter...")
        pipe.load_ip_adapter(
            ip_adapter, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
        )
        pipe.set_ip_adapter_scale(0.8)
    except Exception as e:
        print(f"[WARN]: IP-Adapter load warning: {e}")

    # Only enable CPU offload if explicitly requested AND running low on VRAM
    # This causes heavy CPU usage so it's disabled by default
    if use_cpu_offload:
        print("[WARN]: CPU offload enabled - this will increase CPU usage")
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    return pipe


def run_style_transfer_with_text(
    content_path,
    style_text,
    prompt,
    output_dir,
    negative_prompt="",
    steps=50,
    style_steps=25,
    max_side=1024,
    use_cpu_offload=False,
):
    """
    Run style transfer using text description to generate style reference.

    Args:
        content_path: Path to content image
        style_text: Text description of desired style
        prompt: Generation prompt for final pass
        output_dir: Directory to save output
        negative_prompt: Negative prompt for generation
        steps: Number of inference steps for final pass
        style_steps: Number of steps for style generation
        max_side: Maximum side length for resizing
        use_cpu_offload: Enable CPU offload (uses more CPU, less VRAM)

    Returns:
        dict: Contains output paths and metadata
    """
    import time

    timestamp = int(time.time())

    # Device setup - prefer GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[INFO]: Using device: {device}, dtype: {dtype}")

    if device == "cuda":
        # Print GPU info
        print(f"[INFO]: GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"[INFO]: VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Step 1: Generate style reference from text
    style_image = generate_style_reference(style_text, style_steps, device, dtype)
    style_ref_path = os.path.join(
        output_dir, f"style_transfer_text_generated_style_{timestamp}.png"
    )
    style_image.save(style_ref_path)
    print(f"[SUCCESS]: Saved generated style reference to: {style_ref_path}")

    # Step 2: Load ControlNet pipeline
    pipe = load_controlnet_pipeline(device, dtype, use_cpu_offload=use_cpu_offload)

    # Step 3: Prepare content image
    print(f"[INFO]: Loading content image: {content_path}")
    content_raw = Image.open(content_path).convert("RGB")
    content_image = resize_smart(content_raw, max_side=max_side)
    target_size = content_image.size

    # Resize generated style to match content
    style_image = style_image.resize(target_size, Image.LANCZOS)
    canny_image = get_canny_image(content_image)

    print(f"[INFO]: Running final pass with ControlNet and IP-Adapter...")
    print(f"[INFO]: Final prompt: '{prompt}'")

    out = pipe(
        prompt,
        image=canny_image,
        ip_adapter_image=style_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        controlnet_conditioning_scale=1.0,
    ).images[0]

    # Step 4: Composite result
    final_composite = composite_result(content_raw, out)

    # Save outputs
    output_path = os.path.join(output_dir, f"style_transfer_text_{timestamp}.png")
    final_composite.save(output_path)

    styled_only_path = os.path.join(
        output_dir, f"style_transfer_text_styled_{timestamp}.png"
    )
    out.save(styled_only_path)

    # Create metadata
    result = {
        "task": "style_transfer_with_text",
        "content_image": content_path,
        "style_text": style_text,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "style_steps": style_steps,
        "generated_style_reference": style_ref_path,
        "output_composite": output_path,
        "output_styled_only": styled_only_path,
        "device": device,
        "cpu_offload_enabled": use_cpu_offload,
    }

    # Save metadata JSON
    json_output_path = os.path.join(output_dir, f"style_transfer_text_{timestamp}.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS]: Done! Saved composite to: {output_path}")
    print(f"[SUCCESS]: Saved styled-only to: {styled_only_path}")
    print(f"[SUCCESS]: Saved metadata to: {json_output_path}")

    # Cleanup
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Style Transfer with Text Prompt (GPU Optimized)"
    )
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--style_text", required=True, help="Text description of style")
    parser.add_argument(
        "--prompt", required=True, help="Generation prompt for final pass"
    )
    parser.add_argument(
        "--output_dir", default="./outputs/images", help="Output directory"
    )
    parser.add_argument("--negative_prompt", default="", help="Negative prompt")
    parser.add_argument(
        "--steps", type=int, default=50, help="Inference steps for final pass"
    )
    parser.add_argument(
        "--style_steps", type=int, default=25, help="Steps for style generation"
    )
    parser.add_argument(
        "--max_side", type=int, default=1024, help="Max side length for resize"
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload (saves VRAM but uses more CPU)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_style_transfer_with_text(
        args.content,
        args.style_text,
        args.prompt,
        args.output_dir,
        args.negative_prompt,
        args.steps,
        args.style_steps,
        args.max_side,
        use_cpu_offload=args.cpu_offload,
    )

