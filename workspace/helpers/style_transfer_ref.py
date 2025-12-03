"""
Style Transfer with Reference Image Helpers - Uses unified model cache
Applies style from a reference image to content image.
"""

from PIL import Image, ImageFilter
import numpy as np
import cv2
import json
import os
import time

from .model_cache import get_model_cache


# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================
def resize_smart(image: Image.Image, max_side: int = 1024) -> Image.Image:
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


def get_canny_image(image: Image.Image) -> Image.Image:
    """Extract Canny edges from image for ControlNet."""
    arr = np.array(image.convert("RGB"))

    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(arr)
            gray_gpu = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_RGB2GRAY)
            canny_detector = cv2.cuda.createCannyEdgeDetector(100, 200)
            edges_gpu = canny_detector.detect(gray_gpu)
            edges = edges_gpu.download()
        else:
            raise RuntimeError("No CUDA devices")
    except Exception:
        edges = cv2.Canny(arr, 100, 200)

    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def composite_result(original: Image.Image, styled: Image.Image) -> Image.Image:
    """Composite styled object onto original background using rembg."""
    from rembg import remove

    print("[INFO]: Segmenting object from original image...")

    target_size = styled.size
    original_resized = original.resize(target_size, Image.LANCZOS)

    session = get_model_cache().get_rembg_session()
    masked_original = remove(original_resized, session=session)

    mask = masked_original.split()[-1]
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=2))

    print("[INFO]: Merging styled object onto original background...")
    final_composite = Image.composite(styled, original_resized, mask_blurred)

    return final_composite


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================
def run_style_transfer_ref(
    content_path: str,
    style_path: str,
    prompt: str,
    output_dir: str,
    negative_prompt: str = "cluttered, complex background, dark background",
    steps: int = 50,
    max_side: int = 1024,
) -> dict:
    """
    Run style transfer using a reference image.

    Args:
        content_path: Path to content image
        style_path: Path to style reference image
        prompt: Generation prompt
        output_dir: Directory to save output
        negative_prompt: Negative prompt for generation
        steps: Number of inference steps
        max_side: Maximum side length for resizing

    Returns:
        dict: Contains output path and metadata
    """
    timestamp = int(time.time())
    cache = get_model_cache()

    print(f"[INFO]: Using device: {cache.device}")

    # Get cached pipeline (shared with style_transfer_text)
    pipe = cache.get_controlnet_pipeline()

    # Load images
    print(f"[INFO]: Loading content image: {content_path}")
    content_raw = Image.open(content_path).convert("RGB")

    print(f"[INFO]: Loading style image: {style_path}")
    style_raw = Image.open(style_path).convert("RGB")

    # Pre-process for generation
    content_image = resize_smart(content_raw, max_side=max_side)
    target_size = content_image.size
    style_image = style_raw.resize(target_size, Image.LANCZOS)
    canny_image = get_canny_image(content_image)

    print(f"[INFO]: Running style transfer with prompt: '{prompt}'...")

    start_time = time.time()
    out = pipe(
        prompt,
        image=canny_image,
        ip_adapter_image=style_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        controlnet_conditioning_scale=1.0,
    ).images[0]
    print(f"[INFO]: Inference took {time.time() - start_time:.1f}s")

    # Composite result
    final_image = composite_result(content_raw, out)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"style_transfer_ref_{timestamp}.png")
    final_image.save(output_path)

    styled_only_path = os.path.join(
        output_dir, f"style_transfer_ref_styled_{timestamp}.png"
    )
    out.save(styled_only_path)

    # Create metadata
    result = {
        "task": "style_transfer_with_image",
        "content_image": content_path,
        "style_image": style_path,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "output_composite": output_path,
        "output_styled_only": styled_only_path,
        "device": cache.device,
    }

    # Save metadata JSON
    json_path = os.path.join(output_dir, f"style_transfer_ref_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["metadata_file"] = json_path

    print(f"\n[SUCCESS]: Saved composite to: {output_path}")
    print(f"[SUCCESS]: Saved styled-only to: {styled_only_path}")

    return result
