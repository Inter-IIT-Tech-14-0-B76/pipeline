"""
Prompt Classifier Helpers - Uses unified model cache
Classifies user prompts into task categories using FLAN-T5.
"""

import json
import os
import time

from .model_cache import get_model_cache


# =============================================================================
# CLASSIFICATION PROMPT
# =============================================================================
CLASSIFICATION_PROMPT = """
You are a classifier for an AI photo editing app. Your task is to classify user editing requests based on the user's instruction and the image content.

**Image Description:**
{image_description}

**User Request:**
"{user_request}"

---

Classify the request into EXACTLY ONE of these categories:

1. **style_transfer** - Applying artistic styles, converting to different art forms (e.g., sketch, anime, oil painting, watercolor, cartoon, vintage photo, cinematic look). This transforms the overall aesthetic while preserving the image structure.

2. **color_grading** - Adjusting colors, tones, lighting, and atmosphere (e.g., brightness, contrast, saturation, warmth, coolness, color filters, color correction, exposure, shadows, highlights, vignette).

3. **manual** - Simple geometric or basic transformations that don't require AI models (e.g., crop, rotate, flip, resize, scale, blur, sharpen, basic filters).

4. **default_mode** - Complex content editing that doesn't fit the above categories, such as:
   - Object removal, addition, or replacement
   - Background changes or removal
   - Inpainting or outpainting
   - Face/person modifications
   - Generative edits requiring content understanding
   - Multi-step or ambiguous tasks
   Only use this category if the request clearly doesn't fit style_transfer, color_grading, or manual.

---

**RULES:**
- Analyze BOTH the image description and user request together.
- Consider the IMAGE CONTENT when making your decision.
- Select the MOST SPECIFIC category that applies.
- If the request involves changing the artistic style or aesthetic appearance → style_transfer
- If the request involves adjusting colors, lighting, or tones → color_grading
- If the request is a simple geometric operation → manual
- Only use default_mode if the task strongly requires content-aware AI editing and doesn't fit the other three.
- Output ONLY the category name with no explanation or extra text.

**Response (one word only):**
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
def classify_prompt(user_prompt: str, image_description: str = "") -> str:
    """
    Classify a user prompt into one of the allowed categories.

    Args:
        user_prompt: Text prompt to classify
        image_description: Description of the image content (optional)

    Returns:
        str: One of: style_transfer, color_grading, manual, default_mode
    """
    cache = get_model_cache()
    model, tokenizer = cache.get_flan_t5()

    # Use image description or default
    img_desc = image_description.strip() if image_description else "No image description provided."
    
    prompt = CLASSIFICATION_PROMPT.format(
        image_description=img_desc,
        user_request=user_prompt.strip()
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generate short output (model is on CPU)
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

    # Sanitize: take last line/word that matches allowed categories
    tokens = [t.strip(" .,\n") for t in resp.split()]
    for t in tokens[::-1]:
        if t in ALLOWED_CATEGORIES:
            return t

    # Fallback: find substring match
    for category in ALLOWED_CATEGORIES:
        if category in resp:
            return category

    # Final fallback to default_mode for complex/ambiguous tasks
    return "default_mode"


def run_prompt_classifier(prompt: str, output_dir: str, image_description: str = "") -> dict:
    """
    Run prompt classification using cached FLAN-T5 model.

    Args:
        prompt: Text prompt to classify
        output_dir: Directory to save output JSON
        image_description: Description of the image content (optional)

    Returns:
        dict: Contains classification result
    """
    timestamp = int(time.time())

    print(f"[INFO]: Classifying prompt: '{prompt}'")
    if image_description:
        print(f"[INFO]: Image description: '{image_description[:100]}...'")
    classification = classify_prompt(prompt, image_description)

    # Create output
    result = {
        "task": "prompt_classifier",
        "input_prompt": prompt,
        "image_description": image_description if image_description else None,
        "classification": classification,
        "category_description": CATEGORY_DESCRIPTIONS.get(classification, ""),
        "all_categories": CATEGORY_DESCRIPTIONS,
    }

    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"prompt_classifier_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["output_file"] = output_path

    print(f"\n[SUCCESS]: Results saved to: {output_path}")
    print(f"[INFO]: Classification: {classification}")

    return result
