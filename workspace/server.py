"""
AI Image Processing Server - Flask API with unified model cache.

All models are loaded once at startup and kept in memory for fast inference.

Usage:
    python server.py --port 8000

Endpoints:
    POST /style-transfer/text  - Style transfer with text description
    POST /style-transfer/ref   - Style transfer with reference image
    POST /color-grading        - AI color grading
    POST /ai-suggestions       - Get editing suggestions for an image
    POST /classify             - Classify a prompt into task category
    POST /sam/segment          - SAM2 image segmentation
    POST /comfy/edit           - ComfyUI image editing workflow
    POST /comfy/remix          - ComfyUI image remix (combine two images)
    POST /comfy/inpaint        - ComfyUI removal/inpaint with LaMa
    GET  /health               - Check if server is ready
    GET  /status               - Get detailed server status
"""

import os
import argparse
import json
import traceback

import torch
from flask import Flask, request, jsonify

# Import unified cache
from helpers.model_cache import get_model_cache

# Import task functions (unchanged)
from helpers.style_transfer_text import run_style_transfer
from helpers.style_transfer_ref import run_style_transfer_ref
from helpers.color_grading import run_color_grading
from helpers.ai_suggestions import run_ai_suggestions

from helpers.prompt_classifier import run_prompt_classifier, classify_prompt

# ComfyUI workflow helpers
from helpers.default import main as run_default_edit
from helpers.remix import remix as run_remix
from helpers.removal_inpaint import removal_inpaint as run_removal_inpaint

# Segmentation (SAM2)
from segmentation_sam2.init import segment_image

app = Flask(__name__)

# Default output directories
DEFAULT_OUTPUT_IMAGES = "./outputs/images"
DEFAULT_OUTPUT_DATA = "./outputs/data"


# =============================================================================
# UTIL
# =============================================================================
def _parse_image_analysis_field(raw):
    """
    Accepts:
      - None
      - dict (already parsed)
      - JSON string
      - file path to a JSON file

    Returns parsed dict or None.
    """
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    # If it's a string, try parse JSON, else try read file
    if isinstance(raw, str):
        raw = raw.strip()
        # try JSON parse
        try:
            return json.loads(raw)
        except Exception:
            pass
        # try file path
        if os.path.exists(raw):
            try:
                with open(raw, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return None
    # unknown format
    return None


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    cache = get_model_cache()
    return jsonify(
        {
            "status": "ready" if cache.all_loaded() else "loading",
            "device": cache.device,
        }
    )


@app.route("/status", methods=["GET"])
def status():
    """Detailed status endpoint."""
    cache = get_model_cache()
    status_data = cache.get_status()

    if getattr(cache, "device", None) == "cuda":
        try:
            status_data["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                ),
                "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 1),
            }
        except Exception:
            status_data["gpu"] = {"error": "failed to query GPU details"}

    status_data["status"] = "ready" if cache.all_loaded() else "partial"
    return jsonify(status_data)


# =============================================================================
# STYLE TRANSFER WITH TEXT ENDPOINT
# =============================================================================
@app.route("/style-transfer/text", methods=["POST"])
def style_transfer_text():
    """Style transfer using text description to generate style."""
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        required = ["content", "style_text", "prompt"]
        missing = [p for p in required if p not in params]
        if missing:
            return jsonify({"error": f"Missing: {missing}"}), 400

        if not os.path.exists(params["content"]):
            return jsonify({"error": "Content file not found"}), 404

        result = run_style_transfer(
            content_path=params["content"],
            style_text=params["style_text"],
            prompt=params["prompt"],
            output_dir=params.get("output_dir", DEFAULT_OUTPUT_IMAGES),
            negative_prompt=params.get("negative_prompt", ""),
            steps=params.get("steps", 50),
            style_steps=params.get("style_steps", 25),
            max_side=params.get("max_side", 1024),
        )

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# STYLE TRANSFER WITH REFERENCE IMAGE ENDPOINT
# =============================================================================
@app.route("/style-transfer/ref", methods=["POST"])
def style_transfer_ref():
    """Style transfer using a reference style image."""
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        required = ["content", "style", "prompt"]
        missing = [p for p in required if p not in params]
        if missing:
            return jsonify({"error": f"Missing: {missing}"}), 400

        if not os.path.exists(params["content"]):
            return jsonify({"error": "Content file not found"}), 404

        if not os.path.exists(params["style"]):
            return jsonify({"error": "Style file not found"}), 404

        neg_prompt = "cluttered, complex background, dark background"
        result = run_style_transfer_ref(
            content_path=params["content"],
            style_path=params["style"],
            prompt=params["prompt"],
            output_dir=params.get("output_dir", DEFAULT_OUTPUT_IMAGES),
            negative_prompt=params.get("negative_prompt", neg_prompt),
            steps=params.get("steps", 50),
            max_side=params.get("max_side", 1024),
        )

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# COLOR GRADING ENDPOINT
# =============================================================================
@app.route("/color-grading", methods=["POST"])
def color_grading():
    """AI-powered color grading with parameter extraction."""
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        if "image" not in params:
            return jsonify({"error": "Missing: image"}), 400

        if not os.path.exists(params["image"]):
            return jsonify({"error": "Image file not found"}), 404

        result = run_color_grading(
            image_path=params["image"],
            output_dir_images=params.get("output_dir_images", DEFAULT_OUTPUT_IMAGES),
            output_dir_data=params.get("output_dir_data", DEFAULT_OUTPUT_DATA),
            prompt=params.get("prompt", ""),
            mode=params.get("mode", "both"),
        )

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# AI SUGGESTIONS ENDPOINT
# =============================================================================
@app.route("/ai-suggestions", methods=["POST"])
def ai_suggestions():
    """Get AI-generated editing suggestions for an image."""
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        if "image" not in params:
            return jsonify({"error": "Missing: image"}), 400

        if not os.path.exists(params["image"]):
            return jsonify({"error": "Image file not found"}), 404

        result = run_ai_suggestions(
            image_path=params["image"],
            output_dir=params.get("output_dir", DEFAULT_OUTPUT_DATA),
        )

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# PROMPT CLASSIFIER ENDPOINT (ENHANCED)
# =============================================================================
@app.route("/classify", methods=["POST"])
def classify():
    """Classify a user prompt into task categories.

    Accepts JSON body with keys:
      - prompt (str)                 REQUIRED
      - quick (bool)                 optional: if true, run quick classify (no file writing)
      - image_description (str)      optional: free-text description of image
      - image_analysis (dict|string) optional: structured slots JSON or path/JSON string
      - output_dir (str)             optional: where to save full result (used when quick=False)
    """
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        if "prompt" not in params:
            return jsonify({"error": "Missing: prompt"}), 400

        prompt_text = params["prompt"]
        image_description = params.get("image_description", "")
        image_analysis_raw = params.get("image_analysis", None)
        image_analysis = _parse_image_analysis_field(image_analysis_raw)

        # Quick classification (no file saving)
        if params.get("quick", False):
            # classify_prompt from enhanced module supports image_description & image_analysis
            classification = classify_prompt(
                user_prompt=prompt_text,
                image_description=image_description,
                image_analysis=image_analysis,
            )
            return jsonify(
                {
                    "prompt": prompt_text,
                    "classification": classification,
                    "image_description": image_description or None,
                    "image_analysis_provided": bool(image_analysis),
                }
            )

        # Full classification with file output (run_prompt_classifier now accepts image_analysis)
        result = run_prompt_classifier(
            prompt=prompt_text,
            output_dir=params.get("output_dir", DEFAULT_OUTPUT_DATA),
            image_description=image_description,
            image_analysis=image_analysis,
        )

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# SAM2 SEGMENTATION ENDPOINT
# =============================================================================
@app.route("/sam/segment", methods=["POST"])
def sam_segment():
    """Segment an image using SAM2 with a single point prompt.

    Expects JSON: { "image": "path/to/img.jpg", "x": 150, "y": 200, "output_dir": "optional" }
    """
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        for k in ("image", "x", "y"):
            if k not in params:
                return jsonify({"error": f"Missing: {k}"}), 400

        image_path = params["image"]
        x = int(params["x"])
        y = int(params["y"])
        output_dir = params.get("output_dir", "outputs/segmentation")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404

        mask, output_vis = segment_image(image_path, x, y, output_dir=output_dir)

        mask_path = os.path.join(output_dir, "mask.png")
        out_path = os.path.join(output_dir, "segmented_output.png")

        return jsonify({"mask": mask_path, "output": out_path})

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# COMFYUI: DEFAULT IMAGE EDIT ENDPOINT
# =============================================================================
@app.route("/comfy/edit", methods=["POST"])
def comfy_edit():
    """Edit an image using ComfyUI default workflow.

    Expects JSON: { "image": "path/to/img.jpg", "prompt": "edit instruction" }
    """
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        for k in ("image", "prompt"):
            if k not in params:
                return jsonify({"error": f"Missing: {k}"}), 400

        image_path = params["image"]
        prompt = params["prompt"]

        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404

        result = run_default_edit(image_path, prompt)

        if result is None:
            return jsonify({"error": "Generation failed"}), 500

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# COMFYUI: IMAGE REMIX ENDPOINT
# =============================================================================
@app.route("/comfy/remix", methods=["POST"])
def comfy_remix():
    """Remix two images using ComfyUI workflow.

    Expects JSON: {
        "image1": "path/to/img1.jpg",
        "image2": "path/to/img2.jpg",
        "prompt": "optional combining instruction"
    }
    """
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        for k in ("image1", "image2"):
            if k not in params:
                return jsonify({"error": f"Missing: {k}"}), 400

        image1_path = params["image1"]
        image2_path = params["image2"]
        prompt = params.get("prompt", "")

        if not os.path.exists(image1_path):
            return jsonify({"error": "Image1 file not found"}), 404

        if not os.path.exists(image2_path):
            return jsonify({"error": "Image2 file not found"}), 404

        result = run_remix(image1_path, image2_path, prompt)

        if result is None:
            return jsonify({"error": "Generation failed"}), 500

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# COMFYUI: REMOVAL/INPAINT ENDPOINT
# =============================================================================
@app.route("/comfy/inpaint", methods=["POST"])
def comfy_inpaint():
    """Remove objects from image using mask and LaMa inpainting.

    Expects JSON: {
        "image": "path/to/input.jpg",
        "mask": "path/to/mask.jpg"
    }
    """
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "Empty request body"}), 400

        for k in ("image", "mask"):
            if k not in params:
                return jsonify({"error": f"Missing: {k}"}), 400

        image_path = params["image"]
        mask_path = params["mask"]

        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404

        if not os.path.exists(mask_path):
            return jsonify({"error": "Mask file not found"}), 404

        result = run_removal_inpaint(image_path, mask_path)

        if result is None:
            return jsonify({"error": "Generation failed"}), 500

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR]: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AI Image Processing Server (enhanced classifier)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--skip-preload", action="store_true", help="Skip preloading models"
    )
    parser.add_argument(
        "--preload-sam-only",
        action="store_true",
        help="Only preload the SAM2 model and exit (useful for fast debugging)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="sam2.1_hiera_large.pt",
        help="Path to SAM2 checkpoint file (can be filename or full path)",
    )
    parser.add_argument(
        "--sam-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 config YAML",
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(DEFAULT_OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(DEFAULT_OUTPUT_DATA, exist_ok=True)

    # Preload models using unified cache
    if args.preload_sam_only:
        print("[INFO]: Preloading SAM2 only (will exit after attempt)")
        try:
            get_model_cache().preload_sam(
                config_path=args.sam_config, checkpoint_path=args.sam_checkpoint
            )
            print("[SUCCESS]: SAM2 preload succeeded")
            return
        except Exception as e:
            print(f"[ERROR]: SAM2 preload failed: {e}")
            traceback.print_exc()
            return

    if not args.skip_preload:
        print("[INFO]: Preloading all models (this may take a while)...")
        get_model_cache().preload_all()

    print("\n" + "=" * 60)
    print(f"SERVER RUNNING ON http://{args.host}:{args.port}")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /style-transfer/text  - Style transfer (text)")
    print("  POST /style-transfer/ref   - Style transfer (reference)")
    print("  POST /color-grading        - AI color grading")
    print("  POST /ai-suggestions       - Get editing suggestions")
    print("  POST /classify             - Classify prompt")
    print("  POST /sam/segment          - SAM2 segmentation")
    print("  POST /comfy/edit           - ComfyUI image edit")
    print("  POST /comfy/remix          - ComfyUI image remix")
    print("  POST /comfy/inpaint        - ComfyUI removal/inpaint")
    print("  GET  /health               - Health check")
    print("  GET  /status               - Detailed status")
    print("\n[INFO]: Models preloaded - inference will be fast!")
    print("[INFO]: Press Ctrl+C to stop\n")

    # Run Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()

