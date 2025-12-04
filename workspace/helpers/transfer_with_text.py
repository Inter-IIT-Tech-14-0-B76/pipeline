"""
Transfer with Text Helpers - Uses unified model cache
Generate a style reference from TEXT using SDXL, then apply it to a content image
using ControlNet + IP-Adapter. Finally, composite the styled object back onto
the original background using rembg.
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
    """
    Composite styled object onto original background using rembg.
    Keeps the original background and only styles the foreground object.
    """
    from rembg import remove

    print("[INFO]: Segmenting object from original image...")

    target_size = styled.size
    original_resized = original.resize(target_size, Image.LANCZOS)

    # Use cached rembg session
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


def run_transfer_with_text(
    content_paths: list,
    style_text: str,
    prompt: str,
    output_dir: str,
    negative_prompt: str = "",
    steps: int = 50,
    style_steps: int = 25,
    max_side: int = 1024,
) -> dict:
    """
    Run transfer with text on multiple content images.

    This generates a style reference from text ONCE and applies it to all
    content images, making it efficient for batch processing.

    Args:
        content_paths: List of paths to content images
        style_text: Text description of desired style
        prompt: Generation prompt for final pass
        output_dir: Directory to save output
        negative_prompt: Negative prompt for generation
        steps: Number of inference steps for final pass
        style_steps: Number of steps for style generation
        max_side: Maximum side length for resizing

    Returns:
        dict: Contains output paths and metadata for all processed images
    """
    timestamp = int(time.time())
    cache = get_model_cache()

    print(f"[INFO]: Using device: {cache.device}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate style reference from text (done ONCE for all images)
    start_time = time.time()
    style_image = generate_style_reference(style_text, style_steps)
    style_gen_time = time.time() - start_time
    print(f"[INFO]: Style generation took {style_gen_time:.1f}s")

    style_ref_path = os.path.join(
        output_dir, f"transfer_with_text_generated_style_{timestamp}.png"
    )
    style_image.save(style_ref_path)
    print(f"[SUCCESS]: Saved generated style reference to: {style_ref_path}")

    # Step 2: Get cached ControlNet pipeline
    pipe = cache.get_controlnet_pipeline()

    # Step 3: Process each content image
    results = []
    for idx, content_path in enumerate(content_paths, start=1):
        print(f"\n[INFO]: Processing ({idx}/{len(content_paths)}): {content_path}")

        if not os.path.exists(content_path):
            print(f"[WARN]: Content file not found, skipping: {content_path}")
            results.append(
                {
                    "content_image": content_path,
                    "error": "File not found",
                }
            )
            continue

        # Load and prepare content image
        content_raw = Image.open(content_path).convert("RGB")
        content_image = resize_smart(content_raw, max_side=max_side)
        target_size = content_image.size

        # Resize style image to match content
        style_for_image = style_image.resize(target_size, Image.LANCZOS)
        canny_image = get_canny_image(content_image)

        print(f"[INFO]: Running ControlNet + IP-Adapter pass...")
        start_time = time.time()
        out = pipe(
            prompt,
            image=canny_image,
            ip_adapter_image=style_for_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            controlnet_conditioning_scale=1.0,
        ).images[0]
        inference_time = time.time() - start_time
        print(f"[INFO]: ControlNet inference took {inference_time:.1f}s")

        # Composite result
        final_composite = composite_result(content_raw, out)

        # Generate unique filename based on original
        from pathlib import Path

        content_stem = Path(content_path).stem
        suffix = Path(content_path).suffix if Path(content_path).suffix else ".png"

        output_path = os.path.join(
            output_dir, f"{content_stem}_transfer_with_text_{timestamp}{suffix}"
        )
        final_composite.save(output_path)

        styled_only_path = os.path.join(
            output_dir, f"{content_stem}_transfer_with_text_styled_{timestamp}{suffix}"
        )
        out.save(styled_only_path)

        results.append(
            {
                "content_image": content_path,
                "output_composite": output_path,
                "output_styled_only": styled_only_path,
                "inference_time": round(inference_time, 2),
            }
        )

        print(f"[SUCCESS]: Saved to: {output_path}")

    # Build final result
    result = {
        "task": "transfer_with_text",
        "style_text": style_text,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "style_steps": style_steps,
        "generated_style_reference": style_ref_path,
        "style_generation_time": round(style_gen_time, 2),
        "device": cache.device,
        "images_processed": len([r for r in results if "error" not in r]),
        "images_failed": len([r for r in results if "error" in r]),
        "results": results,
    }

    # Save metadata JSON
    json_output_path = os.path.join(output_dir, f"transfer_with_text_{timestamp}.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["metadata_file"] = json_output_path

    print(f"\n[SUCCESS]: Processed {result['images_processed']} images")
    print(f"[SUCCESS]: Metadata saved to: {json_output_path}")

    return result


# =============================================================================
# SINGLE IMAGE CONVENIENCE FUNCTION
# =============================================================================
def run_transfer_with_text_single(
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
    Convenience function for single image transfer with text.

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
    result = run_transfer_with_text(
        content_paths=[content_path],
        style_text=style_text,
        prompt=prompt,
        output_dir=output_dir,
        negative_prompt=negative_prompt,
        steps=steps,
        style_steps=style_steps,
        max_side=max_side,
    )

    # Flatten result for single image case
    if result["results"] and "error" not in result["results"][0]:
        result["output_composite"] = result["results"][0]["output_composite"]
        result["output_styled_only"] = result["results"][0]["output_styled_only"]
        result["content_image"] = content_path

    return result

