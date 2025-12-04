"""Generate a style reference from TEXT using SDXL, then apply it to a content image using ControlNet + IP-Adapter.
Finally, composite the styled object back onto the original background using rembg.
"""

import argparse
import sys
from pathlib import Path
import torch
from PIL import Image, ImageFilter
import numpy as np

# Try to import rembg
try:
    from rembg import remove
except ImportError:
    print("❌ Error: 'rembg' library is missing.")
    print("   Please install it using: pip install rembg")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--content", required=True, nargs="+", help="Path(s) to content image(s)"
    )
    p.add_argument(
        "--style_text",
        required=True,
        help="Text description of the style to hallucinate",
    )
    p.add_argument("--prompt", required=True, help="Generation prompt for final pass")
    p.add_argument(
        "--negative_prompt", default="", help="Negative prompt for final pass"
    )
    p.add_argument(
        "--output_dir", default="output", help="Output directory to save styled images"
    )
    p.add_argument(
        "--sdxl_model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL model id",
    )
    p.add_argument(
        "--controlnet",
        default="diffusers/controlnet-canny-sdxl-1.0",
        help="ControlNet model id",
    )
    p.add_argument(
        "--ip_adapter", default="h94/IP-Adapter", help="IP-Adapter repo id (optional)"
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detect if omitted)",
    )
    p.add_argument(
        "--steps", type=int, default=50, help="Inference steps for final pass"
    )
    p.add_argument(
        "--style_steps",
        type=int,
        default=25,
        help="Inference steps for style generation from text",
    )
    p.add_argument(
        "--max_side", type=int, default=1024, help="Smart resize max side length"
    )
    return p.parse_args()


def resize_smart(image, max_side=1024):
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
    import cv2

    arr = np.array(image.convert("RGB"))
    edges = cv2.Canny(arr, 100, 200)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)


def composite_result(original, styled):
    """
    Applies the user's custom compositing logic using rembg.
    Keeps the original background and only styles the foreground object.
    """
    print("✂️  Segmenting object from original image (using rembg)...")

    # 1. Resize Original to Match Styled Output EXACTLY
    target_size = styled.size
    original_resized = original.resize(target_size, Image.LANCZOS)

    # 2. Create Mask from ORIGINAL Image
    # 'remove' returns the object on a transparent background
    masked_original = remove(original_resized)

    # 3. Extract the Alpha channel (The Mask)
    # White = Object, Black = Background
    mask = masked_original.split()[-1]

    # 4. Refine Mask (Feathering)
    # Blends edges to avoid the "bad photoshop" look
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=2))

    # 5. Composite
    # Logic: Paste Styled Image ON TOP of Original Image, using the Mask.
    print("✨ Merging styled object onto original background...")
    final_composite = Image.composite(styled, original_resized, mask_blurred)

    return final_composite


def generate_style_reference(text_prompt, sdxl_model, steps, device):
    from diffusers import StableDiffusionXLPipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"🎨 Generating style reference for: '{text_prompt}'")

    # Load separate pipeline for text-to-image
    temp_pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if device == "cuda":
        temp_pipe = temp_pipe.to("cuda")

    img = temp_pipe(prompt=text_prompt, num_inference_steps=steps).images[0]

    # Cleanup to save VRAM
    try:
        del temp_pipe
        torch.cuda.empty_cache()
    except Exception:
        pass
    return img


def main():
    args = parse_args()
    # Normalize content path list
    content_paths = [Path(p) for p in args.content]

    # Filter out non-existent inputs early with a warning
    missing = [str(p) for p in content_paths if not p.exists()]
    if missing:
        for m in missing:
            print(f"Content file not found, skipping: {m}")
    # Keep only existing ones
    content_paths = [p for p in content_paths if p.exists()]
    if len(content_paths) == 0:
        print("No valid content files provided — exiting.")
        return

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Lazy imports for diffusers
    try:
        from diffusers.utils import load_image
    except Exception as e:
        print("Missing diffusers or related packages.")
        raise

    dtype = torch.float16 if device == "cuda" else torch.float32

    # Step A: Hallucinate style image from text (do once for all inputs)
    style_image = generate_style_reference(
        args.style_text, args.sdxl_model, args.style_steps, device
    )
    style_image.save("generated_style_reference.png")
    print("Saved generated style reference to generated_style_reference.png")

    # Step B: Initialize or retrieve cached pipeline
    try:
        pipelines = get_pipelines()
        print("✅ Using cached pipelines from memory")
    except RuntimeError:
        print(
            "⏳ Loading pipeline components for the first time (this may take a few minutes)..."
        )
        pipelines = init_pipelines(
            sdxl_model=args.sdxl_model,
            controlnet_model=args.controlnet,
            ip_adapter=args.ip_adapter,
            device=device,
            dtype=dtype,
        )
        print("✅ Pipelines loaded and cached in process memory")

    # Extract the pipeline object
    pipe = pipelines["pipe"]

    # Step C: Prepare Content
    # Ensure output dir exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each content image
    for idx, content_path in enumerate(content_paths, start=1):
        print(f"\nProcessing ({idx}/{len(content_paths)}): {content_path}")

        content_raw = (
            load_image(str(content_path))
            if "load_image" in globals()
            else Image.open(content_path).convert("RGB")
        )
        content_image = resize_smart(content_raw, max_side=args.max_side)
        target_size = content_image.size

        # Resize generated style image to match content aspect ratio for this image
        style_for_image = style_image.resize(target_size, Image.LANCZOS)
        canny_image = get_canny_image(content_image)

        print("🎨 Running final pass with ControlNet and IP-Adapter...")
        out = pipe(
            args.prompt,
            image=canny_image,
            ip_adapter_image=style_for_image,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            controlnet_conditioning_scale=1.0,
        ).images[0]

        # Background Removal & Composition
        final_composite = composite_result(content_raw, out)

        # Save with consistent naming: <original_stem>_styled<ext>
        suffix = content_path.suffix if content_path.suffix else ".png"
        out_name = f"{content_path.stem}_styled{suffix}"
        out_path = out_dir / out_name
        final_composite.save(out_path)
        print(f"✅ Done — saved to {out_path}")


if __name__ == "__main__":
    main()
