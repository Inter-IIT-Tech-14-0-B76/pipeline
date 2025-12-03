"""
AI Suggestions Helpers - Uses unified model cache
Generates editing prompts for images using moondream2 model.
"""

from PIL import Image
import re
import json
import os
import time

from .model_cache import get_model_cache


# =============================================================================
# QUERY PROMPT
# =============================================================================
AI_SUGGESTIONS_QUERY = """
You are an expert photo editor and creative prompt engineer for AI image-editing tools.

You are given a single image. Analyze it carefully and propose EDITING PROMPTS.

Requirements:

1. IMAGE-SPECIFIC ANALYSIS
   - Briefly describe the image in 1-2 sentences.

2. EDITING SUGGESTIONS AS PROMPTS
   - Propose 6-10 short, ready-to-use editing prompts.
   - Each prompt must start with a verb like: Add, Remove, Replace, Change, Enhance.
   - Make them specific to THIS image.

3. VISUAL ONLY
   - Only describe visual edits to the still image.

4. OUTPUT FORMAT:

   IMAGE_SUMMARY:
   - <1-2 bullet points describing the image>

   EDIT_PROMPTS:
   1. <editing prompt 1>
   2. <editing prompt 2>
   3. <editing prompt 3>
   4. <editing prompt 4>
   5. <editing prompt 5>
   6. <editing prompt 6>
   7. <editing prompt 7>
   8. <editing prompt 8>
   9. <editing prompt 9>
   10. <editing prompt 10>

Only output IMAGE_SUMMARY and EDIT_PROMPTS.
"""


# =============================================================================
# INFERENCE FUNCTION
# =============================================================================
def run_ai_suggestions(image_path: str, output_dir: str) -> dict:
    """
    Run AI suggestions using cached moondream2 model.

    Args:
        image_path: Path to input image
        output_dir: Directory to save output JSON

    Returns:
        dict: Contains image_summary and edit_prompts
    """
    timestamp = int(time.time())
    cache = get_model_cache()

    model, tokenizer = cache.get_moondream()

    print(f"[INFO]: Loading image from {image_path}...")
    image = Image.open(image_path)

    print("[INFO]: Running model inference...")
    raw_output = model.answer_question(
        image, AI_SUGGESTIONS_QUERY, tokenizer, device=cache.device
    )

    # Parse output
    summary_match = re.search(
        r"IMAGE_SUMMARY\s*:?(.+?)EDIT_PROMPTS", raw_output, re.S | re.I
    )
    if summary_match:
        image_summary = summary_match.group(1).strip()
        image_summary = re.sub(r"^[\-\s]+", "", image_summary).strip()
    else:
        image_summary = ""

    prompts_section = raw_output.split("EDIT_PROMPTS:")[-1].strip()
    edit_prompts = re.findall(r"\d+\.\s*(.+)", prompts_section)
    edit_prompts = [p.strip() for p in edit_prompts]

    # Create output
    result = {
        "task": "ai_suggestions",
        "input_image": image_path,
        "image_summary": image_summary,
        "edit_prompts": edit_prompts,
        "raw_output": raw_output,
    }

    # Save to JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"ai_suggestions_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["output_file"] = output_path

    print(f"\n[SUCCESS]: Results saved to: {output_path}")
    print(f"[INFO]: IMAGE SUMMARY: {image_summary}")
    print(f"[INFO]: EDIT PROMPTS ({len(edit_prompts)} found):")
    for i, prompt in enumerate(edit_prompts, 1):
        print(f"  {i}. {prompt}")

    return result
