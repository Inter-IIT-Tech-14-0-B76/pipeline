"""
Color Grading Helpers - Uses unified model cache
AI-powered color grading with parameter extraction using Mask2Former.
"""

import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import json
import os
import time

from .model_cache import get_model_cache


# =============================================================================
# SEGMENTATION FUNCTIONS
# =============================================================================
def get_all_objects(image_path: str) -> tuple:
    """Performs panoptic segmentation to identify all objects in the image."""
    cache = get_model_cache()
    processor, model = cache.get_mask2former()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(cache.device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    panoptic_seg = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]

    found_objects = {}

    for segment in segments_info:
        label_id = segment["label_id"]
        mask_id = segment["id"]
        label_name = model.config.id2label[label_id]

        binary_mask = panoptic_seg == mask_id

        if label_name not in found_objects:
            found_objects[label_name] = binary_mask
        else:
            found_objects[label_name] = np.logical_or(
                found_objects[label_name], binary_mask
            )

    return image, found_objects


def get_grading_recipe(label_name: str) -> str:
    """Maps any COCO label to a specific Grading Style recipe."""
    label = label_name.lower()

    flora_keywords = [
        "tree",
        "grass",
        "bush",
        "plant",
        "flower",
        "broccoli",
        "vegetable",
        "fruit",
    ]
    if any(x in label for x in flora_keywords):
        return "BIO_FLORA"

    water_keywords = ["sky", "cloud", "water", "sea", "river", "lake", "ocean", "pool"]
    if any(x in label for x in water_keywords):
        return "HYDRO_AERO"

    fauna_keywords = [
        "person",
        "face",
        "dog",
        "cat",
        "bird",
        "sheep",
        "cow",
        "horse",
        "food",
        "sandwich",
    ]
    if any(x in label for x in fauna_keywords):
        return "BIO_FAUNA"

    return "STRUCTURAL"


# =============================================================================
# GRADING FUNCTIONS
# =============================================================================
def apply_universal_grade(image_pil: Image.Image, object_masks: dict) -> np.ndarray:
    """Applies color grading adjustments based on identified object masks."""
    img_np = np.array(image_pil).astype(float)
    h, w, _ = img_np.shape

    print("[INFO]: Grading Report:")

    for label, binary_mask in object_masks.items():
        recipe = get_grading_recipe(label)
        print(f"   -> Found '{label}': Applying recipe [{recipe}]")

        mask_u8 = binary_mask.astype(np.uint8) * 255
        blur_amount = 45 if np.sum(binary_mask) > (h * w * 0.1) else 15
        mask_blurred = cv2.GaussianBlur(mask_u8, (blur_amount, blur_amount), 0)
        mask = mask_blurred[:, :, None] / 255.0

        layer = img_np.copy()

        if recipe == "BIO_FLORA":
            layer[:, :, 1] *= 1.20
            layer[:, :, 0] *= 0.90
            layer = (layer - 128) * 1.1 + 128

        elif recipe == "HYDRO_AERO":
            layer[:, :, 0] *= 0.85
            layer[:, :, 2] *= 1.10
            layer = (layer - 128) * 1.15 + 128

        elif recipe == "BIO_FAUNA":
            layer[:, :, 0] *= 1.08
            layer[:, :, 2] *= 0.95
            layer = 255 * (layer / 255) ** 0.9

        elif recipe == "STRUCTURAL":
            mean_val = np.mean(layer)
            layer = (layer - mean_val) * 1.25 + mean_val
            hsv = cv2.cvtColor(
                np.clip(layer, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV
            ).astype(float)
            hsv[:, :, 1] *= 0.9
            layer = cv2.cvtColor(
                np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB
            ).astype(float)

        img_np = img_np * (1 - mask) + layer * mask

    return np.clip(img_np, 0, 255).astype(np.uint8)


def extract_parameters(image_path: str) -> tuple:
    """
    Runs segmentation and calculates global slider parameters
    based on the weighted area of identified objects.
    """
    original, found_objects = get_all_objects(image_path)

    total_pixels = original.size[0] * original.size[1]
    recipe_weights = {}

    for label, mask in found_objects.items():
        recipe = get_grading_recipe(label)
        pixel_count = np.sum(mask)
        weight = pixel_count / total_pixels

        if recipe not in recipe_weights:
            recipe_weights[recipe] = 0
        recipe_weights[recipe] += weight

    print("[INFO]: Image Composition:")
    sorted_weights = sorted(recipe_weights.items(), key=lambda x: x[1], reverse=True)
    for recipe, weight in sorted_weights:
        print(f"   {recipe}: {weight * 100:.1f}%")

    recipe_coeffs = {
        "BIO_FLORA": {
            "r": 0.90,
            "g": 1.20,
            "b": 1.0,
            "contrast": 1.1,
            "sat": 1.0,
            "gamma": 1.0,
        },
        "HYDRO_AERO": {
            "r": 0.85,
            "g": 1.0,
            "b": 1.10,
            "contrast": 1.15,
            "sat": 1.0,
            "gamma": 1.0,
        },
        "BIO_FAUNA": {
            "r": 1.08,
            "g": 1.0,
            "b": 0.95,
            "contrast": 1.0,
            "sat": 1.0,
            "gamma": 0.9,
        },
        "STRUCTURAL": {
            "r": 1.0,
            "g": 1.0,
            "b": 1.0,
            "contrast": 1.25,
            "sat": 0.9,
            "gamma": 1.0,
        },
    }

    weighted = {"r": 0, "g": 0, "b": 0, "contrast": 0, "sat": 0, "gamma": 0}

    for recipe, weight in recipe_weights.items():
        coeffs = recipe_coeffs[recipe]
        for k in weighted:
            weighted[k] += coeffs[k] * weight

    if sum(recipe_weights.values()) == 0:
        weighted = {"r": 1, "g": 1, "b": 1, "contrast": 1, "sat": 1, "gamma": 1}

    gamma_val = weighted["gamma"]
    gamma_param = -np.log2(gamma_val) * 100 if gamma_val > 0 else 0

    params = {
        "exposure": 0.0,
        "contrast": round((weighted["contrast"] - 1.0) * 100, 2),
        "saturation": round((weighted["sat"] - 1.0) * 100, 2),
        "temperature": round((weighted["r"] - weighted["b"]) * 100, 2),
        "tint": round((weighted["g"] - 1.0) * 100, 2),
        "gamma": round(gamma_param, 2),
    }

    print("[INFO]: EXTRACTED PARAMETERS:")
    for k, v in params.items():
        print(f"   {k.capitalize():12s}: {v:+.2f}")

    return original, found_objects, params


def apply_manual_grade(image_path: str, params: dict) -> np.ndarray:
    """Applies manual 'Lightroom-style' slider adjustments to an image."""
    print(f"[INFO]: Applying Manual Grading with params: {params}")

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(float) / 255.0

    # EXPOSURE
    exp_val = params.get("exposure", 0) / 50.0
    img_np = img_np * (2**exp_val)

    # TEMPERATURE
    temp_val = params.get("temperature", 0) / 100.0
    img_np[:, :, 0] *= 1 + temp_val
    img_np[:, :, 2] *= 1 - temp_val

    # TINT
    tint_val = params.get("tint", 0) / 100.0
    img_np[:, :, 1] *= 1 + tint_val

    img_np = np.clip(img_np, 0, 1)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

    # CONTRAST
    contrast_val = 1.0 + (params.get("contrast", 0) / 100.0)
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(contrast_val)

    # SATURATION
    sat_val = 1.0 + (params.get("saturation", 0) / 100.0)
    enhancer = ImageEnhance.Color(img_pil)
    img_pil = enhancer.enhance(sat_val)

    # GAMMA
    gamma_param = params.get("gamma", 0)
    gamma_val = 1.0 - (gamma_param / 200.0)

    img_np = np.array(img_pil).astype(float) / 255.0
    img_np = np.power(img_np, gamma_val)

    img_np = np.clip(img_np, 0, 1) * 255.0
    return img_np.astype(np.uint8)


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================
def run_color_grading(
    image_path: str,
    output_dir_images: str,
    output_dir_data: str,
    prompt: str = "",
    mode: str = "both",
) -> dict:
    """
    Run color grading on an image.

    Args:
        image_path: Path to input image
        output_dir_images: Directory to save output images
        output_dir_data: Directory to save JSON metadata
        prompt: Text prompt (can influence mode selection)
        mode: 'ai' (auto grading), 'manual' (with extracted params), or 'both'

    Returns:
        dict: Contains output paths and extracted parameters
    """
    timestamp = int(time.time())

    # Determine mode from prompt if needed
    if prompt:
        prompt_lower = prompt.lower()
        manual_keywords = ["manual", "slider", "adjust", "increase", "decrease"]
        if any(word in prompt_lower for word in manual_keywords):
            mode = "manual"

    result = {
        "task": "color_grading",
        "input_image": image_path,
        "prompt": prompt,
        "mode": mode,
        "outputs": {},
    }

    # Extract parameters
    print("\n" + "=" * 60)
    print("EXTRACTING COLOR GRADING PARAMETERS")
    print("=" * 60)

    original_img, detected_objects, ai_params = extract_parameters(image_path)

    result["extracted_parameters"] = ai_params
    result["detected_objects"] = list(detected_objects.keys())

    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_data, exist_ok=True)

    # Apply AI grading
    if mode in ["ai", "both"]:
        print("\n" + "=" * 60)
        print("APPLYING AI GRADING")
        print("=" * 60)

        ai_graded_img = apply_universal_grade(original_img, detected_objects)
        ai_output_path = os.path.join(
            output_dir_images, f"color_grading_ai_{timestamp}.png"
        )
        Image.fromarray(ai_graded_img).save(ai_output_path)
        result["outputs"]["ai_graded"] = ai_output_path
        print(f"\n[SUCCESS]: AI graded image saved: {ai_output_path}")

    # Apply manual grading
    if mode in ["manual", "both"]:
        print("\n" + "=" * 60)
        print("APPLYING MANUAL GRADING")
        print("=" * 60)

        manual_result = apply_manual_grade(image_path, ai_params)
        manual_output_path = os.path.join(
            output_dir_images, f"color_grading_manual_{timestamp}.png"
        )
        Image.fromarray(manual_result).save(manual_output_path)
        result["outputs"]["manual_graded"] = manual_output_path
        print(f"\n[SUCCESS]: Manual graded image saved: {manual_output_path}")

    # Save metadata JSON
    json_output_path = os.path.join(output_dir_data, f"color_grading_{timestamp}.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["metadata_file"] = json_output_path

    print("\n" + "=" * 60)
    print("COLOR GRADING COMPLETE")
    print("=" * 60)

    return result
