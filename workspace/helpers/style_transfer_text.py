"""
Style Transfer Helpers - Uses unified model cache
Style transfer with text description to generate style reference.
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
# INFERENCE FUNCTIONS
# =============================================================================
def generate_style_reference(text_prompt: str, steps: int = 25) -> Image.Image:
    """Generate a style reference image from text using cached SDXL pipeline."""
    print(f"[INFO]: Generating style reference for: '{text_prompt}'")

    pipe = get_model_cache().get_style_pipeline()
    img = pipe(prompt=text_prompt, num_inference_steps=steps).images[0]

    return img


def run_style_transfer(
    content_path: str,
    style_text: str,
    prompt: str,
    output_dir: str,
    negative_prompt: str = "",
    steps: int = 50,
    style_steps: int = 25,
    max_side: int = 1024,
) -> dict:
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

    Returns:
        dict: Contains output paths and metadata
    """
    timestamp = int(time.time())
    cache = get_model_cache()

    print(f"[INFO]: Using device: {cache.device}")

    # Step 1: Generate style reference from text
    start_time = time.time()
    style_image = generate_style_reference(style_text, style_steps)
    print(f"[INFO]: Style generation took {time.time() - start_time:.1f}s")

    style_ref_path = os.path.join(
        output_dir, f"style_transfer_text_generated_style_{timestamp}.png"
    )
    style_image.save(style_ref_path)
    print(f"[SUCCESS]: Saved generated style reference to: {style_ref_path}")

    # Step 2: Get cached ControlNet pipeline
    pipe = cache.get_controlnet_pipeline()

    # Step 3: Prepare content image
    print(f"[INFO]: Loading content image: {content_path}")
    content_raw = Image.open(content_path).convert("RGB")
    content_image = resize_smart(content_raw, max_side=max_side)
    target_size = content_image.size

    style_image = style_image.resize(target_size, Image.LANCZOS)
    canny_image = get_canny_image(content_image)

    print(f"[INFO]: Running ControlNet + IP-Adapter pass...")
    print(f"[INFO]: Prompt: '{prompt}'")

    start_time = time.time()
    out = pipe(
        prompt,
        image=canny_image,
        ip_adapter_image=style_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        controlnet_conditioning_scale=1.0,
    ).images[0]
    print(f"[INFO]: ControlNet inference took {time.time() - start_time:.1f}s")

    # Step 4: Composite result
    final_composite = composite_result(content_raw, out)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"style_transfer_text_{timestamp}.png")
    final_composite.save(output_path)

    styled_only_path = os.path.join(
        output_dir, f"style_transfer_text_styled_{timestamp}.png"
    )
    out.save(styled_only_path)

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
        "device": cache.device,
    }

    json_output_path = os.path.join(output_dir, f"style_transfer_text_{timestamp}.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS]: Saved composite to: {output_path}")
    print(f"[SUCCESS]: Saved styled-only to: {styled_only_path}")

    return result
