"""
prompt_classifier_enhanced.py

Enhanced prompt classifier for an AI photo editing app.
Accepts:
 - user_prompt (str)
 - image_description (optional str)
 - image_analysis (optional dict) : structured slots from image analysis (subject, style, constraints, etc.)

Uses cached FLAN-T5 model via get_model_cache(). Behavior and outputs are similar to the original script,
but classification prompt now includes the structured image analysis for better context.
"""

import json
import os
import time
from typing import Any, Dict, Optional

from .model_cache import get_model_cache  # unchanged dependency from original script

# =============================================================================
# CLASSIFICATION PROMPT
# =============================================================================
CLASSIFICATION_PROMPT = """
You are a classifier for an AI photo editing app. Your task is to classify user editing requests based on the user's instruction and the image content.

Image Description (natural language):
{image_description}

Image Analysis (structured slots JSON):
{image_slots_json}

User Request:
"{user_request}"

Classify the request into EXACTLY ONE of these categories (OUTPUT ONE WORD ONLY; no explanation):

1. style_transfer - Applying artistic styles, converting to different art forms (sketch, anime, oil painting, watercolor, cartoon, vintage photo, cinematic look). Transforms aesthetic while preserving structure.

2. color_grading - Adjusting colors, tones, lighting, and atmosphere (brightness, contrast, saturation, warmth/coolness, color filters, exposure, shadows, highlights, vignette).

3. manual - Simple geometric or basic transformations that don't require AI models (crop, rotate, flip, resize, scale, blur, sharpen, simple filter).

4. default_mode - Complex content-editing requiring content understanding or generative editing: object removal/addition/replacement, background changes/removal, inpainting/outpainting, face/person modifications, multi-step or ambiguous tasks.

Rules:
- Use BOTH the image description and image analysis slots together with the user request.
- Choose the MOST SPECIFIC category that applies.
- If the request changes artistic style or overall aesthetic -> style_transfer.
- If the request adjusts color, lighting, or tone -> color_grading.
- If the request is basic geometric or single-step filter-like -> manual.
- Use default_mode only if the task clearly requires content-aware or generative editing.

Response (one word only):
"""

ALLOWED_CATEGORIES = {"style_transfer", "color_grading", "manual", "default_mode"}

CATEGORY_DESCRIPTIONS = {
    "style_transfer": "Artistic style transformation (sketch, anime, oil painting, etc.)",
    "color_grading": "Color/tone/lighting adjustments (brightness, contrast, filters, etc.)",
    "manual": "Simple geometric operations (crop, rotate, flip, resize, blur)",
    "default_mode": "Complex content-aware editing (object removal/addition, background changes, inpainting, etc.)",
}


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================
def _format_image_slots(slots: Optional[Dict[str, Any]]) -> str:
    """Return a pretty JSON string for image slots (or a placeholder)."""
    if not slots:
        return "No structured image analysis provided."
    try:
        return json.dumps(slots, indent=2, ensure_ascii=False)
    except Exception:
        # fallback: simple string representation
        return str(slots)


def classify_prompt(
    user_prompt: str,
    image_description: str = "",
    image_analysis: Optional[Dict[str, Any]] = None,
    max_length: int = 512,
) -> str:
    """
    Classify a user prompt into one of the allowed categories.

    Args:
        user_prompt: Text prompt to classify
        image_description: Optional natural-language description of the image
        image_analysis: Optional dict containing structured slots (subject, style, constraints, ...)
        max_length: tokenization truncation max length (passed to tokenizer)

    Returns:
        str: One of: style_transfer, color_grading, manual, default_mode
    """
    cache = get_model_cache()
    model, tokenizer = cache.get_flan_t5()

    img_desc = (
        image_description.strip()
        if image_description
        else "No image description provided."
    )
    img_slots_json = _format_image_slots(image_analysis)

    prompt = CLASSIFICATION_PROMPT.format(
        image_description=img_desc,
        image_slots_json=img_slots_json,
        user_request=user_prompt.strip(),
    )

    # Tokenize (short input -> small max_length to keep generation short)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    )

    # Generate short output (model may be on CPU)
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resp = resp.strip().lower()

    # Extract the last token/word that matches allowed categories
    # split by whitespace and punctuation
    tokens = [t.strip(" .,\n\"'") for t in resp.split()]
    for t in tokens[::-1]:
        if t in ALLOWED_CATEGORIES:
            return t

    # substring match fallback
    for category in ALLOWED_CATEGORIES:
        if category in resp:
            return category

    # As final safety fallback, attempt simple heuristics on user_prompt & image_analysis
    combined = (
        user_prompt
        + " "
        + (image_description or "")
        + " "
        + json.dumps(image_analysis or {})
    ).lower()
    if any(
        w in combined
        for w in [
            "paint",
            "watercolor",
            "oil painting",
            "anime",
            "cartoon",
            "photorealistic",
            "style",
        ]
    ):
        return "style_transfer"
    if any(
        w in combined
        for w in [
            "color",
            "brightness",
            "contrast",
            "saturation",
            "exposure",
            "tone",
            "white balance",
            "warm",
            "cool",
        ]
    ):
        return "color_grading"
    if any(
        w in combined
        for w in ["crop", "rotate", "flip", "resize", "blur", "sharpen", "scale"]
    ):
        return "manual"

    return "default_mode"


def run_prompt_classifier(
    prompt: str,
    output_dir: str,
    image_description: str = "",
    image_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run prompt classification using cached FLAN-T5 model.

    Args:
        prompt: Text prompt to classify
        output_dir: Directory to save output JSON
        image_description: Optional natural-language image description
        image_analysis: Optional structured slots dictionary (from image analysis)

    Returns:
        dict: Contains classification result and metadata (same structure as original, with image_analysis added)
    """
    timestamp = int(time.time())

    print(f"[INFO]: Classifying prompt: '{prompt}'")
    if image_description:
        print(f"[INFO]: Image description: '{image_description[:200]}...'")
    if image_analysis:
        print(f"[INFO]: Image analysis slots provided: {list(image_analysis.keys())}")

    classification = classify_prompt(
        prompt, image_description=image_description, image_analysis=image_analysis
    )

    result = {
        "task": "prompt_classifier",
        "input_prompt": prompt,
        "image_description": image_description if image_description else None,
        "image_analysis": image_analysis if image_analysis else None,
        "classification": classification,
        "category_description": CATEGORY_DESCRIPTIONS.get(classification, ""),
        "all_categories": CATEGORY_DESCRIPTIONS,
        "timestamp": timestamp,
    }

    # Ensure output dir exists and write JSON
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"prompt_classifier_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["output_file"] = output_path

    print(f"\n[SUCCESS]: Results saved to: {output_path}")
    print(f"[INFO]: Classification: {classification}")

    return result


# If run as a script for quick local testing (keeps behavior similar to original)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prompt Classifier (enhanced with image analysis slots)"
    )
    parser.add_argument(
        "--prompt", "-p", required=True, help="User editing prompt to classify"
    )
    parser.add_argument(
        "--image_description",
        "-d",
        default="",
        help="Optional natural-language image description",
    )
    parser.add_argument(
        "--image_analysis",
        "-s",
        default="",
        help="Optional path to JSON file containing structured image slots (or raw JSON string)",
    )
    parser.add_argument(
        "--out", "-o", default=".", help="Output directory to save result JSON"
    )
    args = parser.parse_args()

    image_analysis_obj = None
    if args.image_analysis:
        # try to load from file first, else parse as JSON string
        try:
            if os.path.exists(args.image_analysis):
                with open(args.image_analysis, "r", encoding="utf-8") as fh:
                    image_analysis_obj = json.load(fh)
            else:
                image_analysis_obj = json.loads(args.image_analysis)
        except Exception:
            print(
                "[warn] Could not parse image_analysis argument as JSON or file; ignoring structured slots."
            )
            image_analysis_obj = None

    run_prompt_classifier(
        prompt=args.prompt,
        output_dir=args.out,
        image_description=args.image_description,
        image_analysis=image_analysis_obj,
    )
