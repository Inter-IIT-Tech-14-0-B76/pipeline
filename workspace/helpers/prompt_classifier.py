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

Number of images provided: {num_images}

User Request:
"{user_request}"

Classify the request into EXACTLY ONE of these categories (OUTPUT ONE WORD ONLY; no explanation):

1. style_transfer - Applying artistic styles using TEXT DESCRIPTION, converting to different art forms (sketch, anime, oil painting, watercolor, cartoon, vintage photo, cinematic look). Transforms aesthetic while preserving structure. Use this when user describes style in words.

2. style_transfer_ref - Copying/transferring style FROM A REFERENCE IMAGE to another image. Use this when user provides TWO IMAGES and wants to apply the style/look of one image onto another. Keywords: "copy style", "match style", "like this image", "transfer style from", "make it look like".

3. color_grading - Adjusting colors, tones, lighting, and atmosphere (brightness, contrast, saturation, warmth/coolness, color filters, exposure, shadows, highlights, vignette).

4. manual - Simple geometric or basic transformations that don't require AI models (crop, rotate, flip, resize, scale, blur, sharpen, simple filter).

5. default_mode - Complex content-editing requiring content understanding or generative editing: object removal/addition/replacement, background changes/removal, inpainting/outpainting, face/person modifications, multi-step or ambiguous tasks.

Rules:
- Use BOTH the image description and image analysis slots together with the user request.
- Choose the MOST SPECIFIC category that applies.
- If TWO IMAGES are provided AND user wants to copy/transfer style between them -> style_transfer_ref.
- If the request changes artistic style using text description -> style_transfer.
- If the request adjusts color, lighting, or tone -> color_grading.
- If the request is basic geometric or single-step filter-like -> manual.
- Use default_mode only if the task clearly requires content-aware or generative editing.

Response (one word only):
"""

ALLOWED_CATEGORIES = {
    "style_transfer",
    "style_transfer_ref",
    "color_grading",
    "manual",
    "default_mode",
}

CATEGORY_DESCRIPTIONS = {
    "style_transfer": "Artistic style transformation using text description (sketch, anime, oil painting, etc.)",
    "style_transfer_ref": "Style transfer using a reference image - copies style from one image to another",
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


def _determine_image_roles(
    user_prompt: str,
    image_analyses: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Determine which image is the style/reference and which is the content image.

    Uses heuristics based on:
    - User prompt keywords (e.g., "first image", "second image", "style of", "apply to")
    - Image analysis metadata (if available)

    Args:
        user_prompt: The user's editing request
        image_analyses: List of image analysis dicts (one per image)

    Returns:
        Dict with 'style_index', 'content_index', and 'confidence' keys
    """
    prompt_lower = user_prompt.lower()

    # Default: first image is style, second is content
    # (common pattern: "apply style of first image to second")
    style_index = 0
    content_index = 1
    confidence = "low"

    # Check for explicit ordering in prompt
    # Patterns where first image is style reference
    first_is_style_patterns = [
        "style of first",
        "first image style",
        "style from first",
        "like the first",
        "first image's style",
        "copy first",
        "transfer from first",
        "first one's style",
        "style of image 1",
        "image 1 style",
        "style from image 1",
    ]

    # Patterns where second image is style reference
    second_is_style_patterns = [
        "style of second",
        "second image style",
        "style from second",
        "like the second",
        "second image's style",
        "copy second",
        "transfer from second",
        "second one's style",
        "style of image 2",
        "image 2 style",
        "style from image 2",
    ]

    # Patterns where first is content (to be styled)
    first_is_content_patterns = [
        "apply to first",
        "first image to",
        "style the first",
        "edit first",
        "transform first",
        "first one to",
        "apply to image 1",
        "image 1 to",
    ]

    # Patterns where second is content (to be styled)
    second_is_content_patterns = [
        "apply to second",
        "second image to",
        "style the second",
        "edit second",
        "transform second",
        "second one to",
        "apply to image 2",
        "image 2 to",
    ]

    for pattern in first_is_style_patterns:
        if pattern in prompt_lower:
            style_index = 0
            content_index = 1
            confidence = "high"
            break

    for pattern in second_is_style_patterns:
        if pattern in prompt_lower:
            style_index = 1
            content_index = 0
            confidence = "high"
            break

    for pattern in first_is_content_patterns:
        if pattern in prompt_lower:
            content_index = 0
            style_index = 1
            confidence = "high"
            break

    for pattern in second_is_content_patterns:
        if pattern in prompt_lower:
            content_index = 1
            style_index = 0
            confidence = "high"
            break

    # If we have image analyses, use metadata hints
    if image_analyses and len(image_analyses) >= 2 and confidence == "low":
        for i, analysis in enumerate(image_analyses):
            if analysis:
                role = analysis.get("role", "").lower()
                img_type = analysis.get("type", "").lower()

                if role in ["style", "reference"] or img_type in ["style", "reference"]:
                    style_index = i
                    content_index = (
                        1 - i
                        if len(image_analyses) == 2
                        else (i + 1) % len(image_analyses)
                    )
                    confidence = "medium"
                    break
                elif role in ["content", "target", "main"] or img_type in [
                    "content",
                    "target",
                    "main",
                ]:
                    content_index = i
                    style_index = (
                        1 - i
                        if len(image_analyses) == 2
                        else (i + 1) % len(image_analyses)
                    )
                    confidence = "medium"
                    break

    return {
        "style_index": style_index,
        "content_index": content_index,
        "confidence": confidence,
    }


def classify_prompt(
    user_prompt: str,
    image_description: str = "",
    image_analysis: Optional[Dict[str, Any]] = None,
    image_analyses: Optional[list] = None,
    num_images: int = 1,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Classify a user prompt into one of the allowed categories.

    Args:
        user_prompt: Text prompt to classify
        image_description: Optional natural-language description of the image(s)
        image_analysis: Optional dict containing structured slots (subject, style, constraints, ...)
        image_analyses: Optional list of image analysis dicts (one per image, for multi-image scenarios)
        num_images: Number of images provided (default 1)
        max_length: tokenization truncation max length (passed to tokenizer)

    Returns:
        Dict containing:
            - classification: str (one of: style_transfer, style_transfer_ref, color_grading, manual, default_mode)
            - image_roles: Dict with style_index, content_index (only for style_transfer_ref)
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
        num_images=num_images,
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

    classification = None

    # Extract the last token/word that matches allowed categories
    # split by whitespace and punctuation
    tokens = [t.strip(" .,\n\"'") for t in resp.split()]
    for t in tokens[::-1]:
        if t in ALLOWED_CATEGORIES:
            classification = t
            break

    # substring match fallback
    if not classification:
        for category in ALLOWED_CATEGORIES:
            if category in resp:
                classification = category
                break

    # As final safety fallback, attempt simple heuristics on user_prompt & image_analysis
    if not classification:
        combined = (
            user_prompt
            + " "
            + (image_description or "")
            + " "
            + json.dumps(image_analysis or {})
        ).lower()

        # Check for style_transfer_ref first (needs multiple images + style transfer intent)
        style_ref_keywords = [
            "copy style",
            "match style",
            "like this image",
            "transfer style",
            "style from",
            "same style as",
            "style of the",
            "apply the style",
            "reference image",
            "style reference",
        ]
        if num_images >= 2 and any(w in combined for w in style_ref_keywords):
            classification = "style_transfer_ref"
        elif any(
            w in combined
            for w in [
                "paint",
                "watercolor",
                "oil painting",
                "anime",
                "cartoon",
                "photorealistic",
            ]
        ):
            classification = "style_transfer"
        elif any(
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
            classification = "color_grading"
        elif any(
            w in combined
            for w in ["crop", "rotate", "flip", "resize", "blur", "sharpen", "scale"]
        ):
            classification = "manual"
        else:
            classification = "default_mode"

    # Build result dict
    result = {"classification": classification}

    # For style_transfer_ref, determine image roles
    if classification == "style_transfer_ref" and num_images >= 2:
        result["image_roles"] = _determine_image_roles(user_prompt, image_analyses)

    return result


def run_prompt_classifier(
    prompt: str,
    output_dir: str,
    image_description: str = "",
    image_analysis: Optional[Dict[str, Any]] = None,
    image_analyses: Optional[list] = None,
    num_images: int = 1,
) -> Dict[str, Any]:
    """
    Run prompt classification using cached FLAN-T5 model.

    Args:
        prompt: Text prompt to classify
        output_dir: Directory to save output JSON
        image_description: Optional natural-language image description
        image_analysis: Optional structured slots dictionary (from image analysis)
        image_analyses: Optional list of image analysis dicts (one per image)
        num_images: Number of images provided (default 1)

    Returns:
        dict: Contains classification result, image_roles (for style_transfer_ref), and metadata
    """
    timestamp = int(time.time())

    print(f"[INFO]: Classifying prompt: '{prompt}'")
    print(f"[INFO]: Number of images: {num_images}")
    if image_description:
        print(f"[INFO]: Image description: '{image_description[:200]}...'")
    if image_analysis:
        print(f"[INFO]: Image analysis slots provided: {list(image_analysis.keys())}")
    if image_analyses:
        print(f"[INFO]: Image analyses provided for {len(image_analyses)} images")

    classification_result = classify_prompt(
        prompt,
        image_description=image_description,
        image_analysis=image_analysis,
        image_analyses=image_analyses,
        num_images=num_images,
    )

    classification = classification_result["classification"]
    image_roles = classification_result.get("image_roles")

    result = {
        "task": "prompt_classifier",
        "input_prompt": prompt,
        "num_images": num_images,
        "image_description": image_description if image_description else None,
        "image_analysis": image_analysis if image_analysis else None,
        "classification": classification,
        "image_roles": image_roles,
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
    if image_roles:
        print(
            f"[INFO]: Image roles: style_index={image_roles['style_index']}, content_index={image_roles['content_index']}"
        )

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
