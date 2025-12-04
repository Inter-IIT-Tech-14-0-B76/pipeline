import json
import urllib.request
import urllib.parse
import urllib.error
import requests  # pip install requests
import time
import argparse
import sys
import os
import random


OUTPUT_FOLDER = "/workspace/comfy_img_remix/comfycode/outputs"
COMFY_URL = "http://0.0.0.0:8001"
WORKFLOW = {
    "1": {
        "inputs": {"strength": 1, "model": ["3", 0]},
        "class_type": "CFGNorm",
        "_meta": {"title": "CFGNorm"},
    },
    "2": {
        "inputs": {"samples": ["10", 0], "vae": ["8", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "3": {
        "inputs": {"shift": 3, "model": ["9", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "ModelSamplingAuraFlow"},
    },
    "4": {
        "inputs": {
            "filename_prefix": "ComfyUI-qwen-2509-gguf-output",
            "images": ["2", 0],
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
    "6": {
        "inputs": {"unet_name": "Qwen-Image-Edit-2509-Q4_0.gguf"},
        "class_type": "UnetLoaderGGUF",
        "_meta": {"title": "Unet Loader (GGUF)"},
    },
    "7": {
        "inputs": {
            "clip_name": "qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image",
            "device": "default",
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"},
    },
    "8": {
        "inputs": {"vae_name": "qwen-image/qwen_image_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "9": {
        "inputs": {
            "lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            "strength_model": 1,
            "model": ["6", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LoraLoaderModelOnly"},
    },
    "10": {
        "inputs": {
            "seed": 950914352603805,
            "steps": 5,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["1", 0],
            "positive": ["15", 0],
            "negative": ["13", 0],
            "latent_image": ["14", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "11": {
        "inputs": {"image": "240_F_1613056173_eRqGcprd65LvaA3yd10ytmGGHc4YQBIc.jpg"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "12": {
        "inputs": {"image": "1000_F_385159962_dPXKOH58Bdeq4MNgxRYGLXygJ6BtROWv.jpg"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "13": {
        "inputs": {
            "prompt": "",
            "clip": ["7", 0],
            "vae": ["8", 0],
            "image1": ["11", 0],
            "image2": ["12", 0],
        },
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "TextEncodeQwenImageEditPlus"},
    },
    "14": {
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
        "class_type": "EmptySD3LatentImage",
        "_meta": {"title": "EmptySD3LatentImage"},
    },
    "15": {
        "inputs": {
            "prompt": "change this to the baby being on the front porch of the house in first image",
            "clip": ["7", 0],
            "vae": ["8", 0],
            "image1": ["11", 0],
            "image2": ["12", 0],
        },
        "class_type": "TextEncodeQwenImageEditPlus",
        "_meta": {"title": "TextEncodeQwenImageEditPlus"},
    },
    "28": {
        "inputs": {"unet_name": "flux1-dev-Q4_0.gguf"},
        "class_type": "UnetLoaderGGUF",
        "_meta": {"title": "Unet Loader (GGUF)"},
    },
    "29": {
        "inputs": {
            "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux",
            "device": "default",
        },
        "class_type": "DualCLIPLoader",
        "_meta": {"title": "DualCLIPLoader"},
    },
    "30": {
        "inputs": {"vae_name": "ae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "32": {
        "inputs": {"text": ["40", 0], "clip": ["29", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "33": {
        "inputs": {
            "seed": 348266546690106,
            "steps": 4,
            "cfg": 0.9,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.2,
            "model": ["28", 0],
            "positive": ["32", 0],
            "negative": ["36", 0],
            "latent_image": ["37", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "34": {
        "inputs": {"samples": ["33", 0], "vae": ["30", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "35": {
        "inputs": {"filename_prefix": "flux_critique_fix", "images": ["34", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
    "36": {
        "inputs": {"text": "", "clip": ["29", 0]},
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "CLIP Text Encode (Prompt)"},
    },
    "37": {
        "inputs": {"pixels": ["2", 0], "vae": ["30", 0]},
        "class_type": "VAEEncode",
        "_meta": {"title": "VAE Encode"},
    },
    "38": {
        "inputs": {
            "preview": "- Correct the lighting on the door to ensure it matches the warm, ambient light of the scene.\n- Adjust the position of the pumpkin to ensure it is centered and properly aligned with the child.\n- Sharpen the details of the child's outfit to make the blue overalls more visible and natural.\n- Enhance the texture of the rug to make it appear more polished and well-defined.\n- Remove any shadows that appear too long or uneven, ensuring they match the lighting conditions.\n- Correct the perspective to ensure the child is clearly visible and the background elements are proportionally balanced.",
            "source": ["40", 0],
        },
        "class_type": "PreviewAny",
        "_meta": {"title": "Preview as Text"},
    },
    "39": {
        "inputs": {
            "model_name": "OpenGVLab/InternVL3_5-1B",
            "quantization": "none",
            "precision": "float16",
            "device": "cuda",
            "auto_download": "enable",
        },
        "class_type": "InternVL3_5_ModelLoader",
        "_meta": {"title": "👁️ Load InternVL3.5 Model"},
    },
    "40": {
        "inputs": {
            "system_prompt": "You are an expert visual editor and AI image-quality inspector.\nYou receive one generated image that is an early attempt at combining two unknown images.\n\nYour task is to look at the image carefully and independently, and identify all specific visual problems that prevent the scene from looking realistic, coherent, or high-quality.\nYou may consider common issues such as:\n\nphysics or motion inconsistencies\nlighting or shadow mismatches\nscale, perspective, or depth errors\nanatomy or facial distortions\nenvironment or emotions mismatch\ntexture or rendering artifacts\n\nBut do NOT restrict yourself to these categories only.\nIf the image contains any unusual or image-specific flaws, inconsistencies, or oddities, you must identify and address them as well.\n\nAfter analyzing the image, convert your findings into a single, concise list of direct action commands that a diffusion model should follow to improve the next generation.\n\nThese commands must:\n\nStart with action verbs (“Fix…”, “Adjust…”, “Remove…”, “Sharpen…”, “Clean…”, “Correct…”, “Align…”)\nBe highly specific to the exact image you see\nDescribe precise improvements (what to change, enhance, remove, or realign)\nAvoid generic suggestions\nAvoid prose, critiques, or explanations\nAvoid any reference to being AI-generated\nAvoid any reference to previous steps or source images\nOutput only the command list. No other sections, no commentary, no analysis text.\n\nExample of GOOD commands for a different, random image (do NOT reuse these literally):\n\nFix the harsh shadow under the woman’s chin so it matches the soft window light direction.\nAdjust the size of the dog so it is slightly smaller and fits naturally next to the sofa.\nCorrect the man’s right hand so all five fingers are clearly separated and anatomically correct.\nAlign the reflections in the glass table with the actual positions of the lamps above it.\nRemove the blurry duplicate outline around the mountain peak in the background.\nSharpen the facial features of the child while keeping the skin texture smooth and natural.\nClean the noisy color artifacts in the dark corners of the room.\nImprove the depth by slightly increasing background blur while keeping the main subject sharp.\n\nNow do the same style of commands, but ONLY for the image you see.",
            "prompt": "You are an expert visual editor and AI image-quality inspector.\nYou receive one generated image that is an early attempt at combining two unknown images.\n\nYour task is to look at the image carefully and independently, and identify all specific visual problems that prevent the scene from looking realistic, coherent, or high-quality.\nYou may consider common issues such as:\n\nphysics or motion inconsistencies\nlighting or shadow mismatches\nscale, perspective, or depth errors\nanatomy or facial distortions\nenvironment or emotions mismatch\ntexture or rendering artifacts\n\nBut do NOT restrict yourself to these categories only.\nIf the image contains any unusual or image-specific flaws, inconsistencies, or oddities, you must identify and address them as well.\n\nAfter analyzing the image, convert your findings into a single, concise list of direct action commands that a diffusion model should follow to improve the next generation.\n\nThese commands must:\n\nStart with action verbs (“Fix…”, “Adjust…”, “Remove…”, “Sharpen…”, “Clean…”, “Correct…”, “Align…”)\nBe highly specific to the exact image you see\nDescribe precise improvements (what to change, enhance, remove, or realign)\nAvoid generic suggestions\nAvoid prose, critiques, or explanations\nAvoid any reference to being AI-generated\nAvoid any reference to previous steps or source images\nOutput only the command list. No other sections, no commentary, no analysis text.\n\nExample of GOOD commands for a different, random image (do NOT reuse these literally):\n\nFix the harsh shadow under the woman’s chin so it matches the soft window light direction.\nAdjust the size of the dog so it is slightly smaller and fits naturally next to the sofa.\nCorrect the man’s right hand so all five fingers are clearly separated and anatomically correct.\nAlign the reflections in the glass table with the actual positions of the lamps above it.\nRemove the blurry duplicate outline around the mountain peak in the background.\nSharpen the facial features of the child while keeping the skin texture smooth and natural.\nClean the noisy color artifacts in the dark corners of the room.\nImprove the depth by slightly increasing background blur while keeping the main subject sharp.\n\nNow do the same style of commands, but ONLY for the image you see.",
            "special_captioning_token": "",
            "seed": 815884360015265,
            "max_num_tiles": 24,
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 50,
            "internvl_model": ["39", 0],
            "image": ["2", 0],
        },
        "class_type": "InternVL3_5_ImageToText",
        "_meta": {"title": "👁️ InternVL3.5 Image to Text"},
    },
    "41": {
        "inputs": {"preview": "", "source": ["40", 1]},
        "class_type": "PreviewAny",
        "_meta": {"title": "Preview as Text"},
    },
    "42": {
        "inputs": {
            "preview": "- Correct the lighting on the door to ensure it matches the warm, ambient light of the scene.\n- Adjust the position of the pumpkin to ensure it is centered and properly aligned with the child.\n- Sharpen the details of the child's outfit to make the blue overalls more visible and natural.\n- Enhance the texture of the rug to make it appear more polished and well-defined.\n- Remove any shadows that appear too long or uneven, ensuring they match the lighting conditions.\n- Correct the perspective to ensure the child is clearly visible and the background elements are proportionally balanced.",
            "source": ["40", 2],
        },
        "class_type": "PreviewAny",
        "_meta": {"title": "Preview as Text"},
    },
    "43": {
        "inputs": {"preview": "", "source": ["40", 3]},
        "class_type": "PreviewAny",
        "_meta": {"title": "Preview as Text"},
    },
}


def upload_image(image_path, type_desc):
    """Uploads an image to ComfyUI and returns the server filename."""
    if not os.path.exists(image_path):
        sys.exit(f"[ERROR]: {type_desc} file not found: {image_path}")

    url = f"{COMFY_URL}/upload/image"
    print(f"[INFO] Uploading {type_desc}: {os.path.basename(image_path)}...", end="")

    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"overwrite": "true"}
        try:
            response = requests.post(url, files=files, data=data)
        except requests.exceptions.ConnectionError:
            sys.exit(
                f"\n[ERROR]: Cannot connect to ComfyUI at {COMFY_URL}. Is it running?"
            )

    if response.status_code == 200:
        server_filename = response.json().get("name")
        print(f" Done! ({server_filename})")
        return server_filename
    else:
        sys.exit(f"\nUpload failed: {response.status_code} - {response.text}")


def queue_prompt(workflow_data):
    p = {"prompt": workflow_data}
    data = json.dumps(p).encode("utf-8")
    try:
        req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        print(f"\n[!] SERVER ERROR {e.code} [!]")
        print(f"Message: {e.read().decode('utf-8')}")
        sys.exit(1)


def get_history(prompt_id):
    with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())


def download_image(filename, subfolder, folder_type):
    """Downloads the generated image to the local OUTPUT_FOLDER."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    try:
        with urllib.request.urlopen(f"{COMFY_URL}/view?{url_values}") as response:
            data = response.read()
            local_path = os.path.join(OUTPUT_FOLDER, filename)
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"   [SAVED] Image saved to: {local_path}")
    except Exception as e:
        print(f"   [ERROR] Could not download image: {e}")


def remix(image1, image2, prompt="", workflow=WORKFLOW):
    server_img1 = upload_image(image1, "Image 1")
    server_img2 = upload_image(image2, "Image 2")

    print("[INFO] Configuring workflow nodes...")

    # Node 11: Input Image 1
    if "11" in workflow:
        workflow["11"]["inputs"]["image"] = server_img1
    else:
        sys.exit("[ERROR]: Node 11 (Load Image) not found.")

    # Node 12: Input Image 2
    if "12" in workflow:
        workflow["12"]["inputs"]["image"] = server_img2
    else:
        sys.exit("[ERROR]: Node 12 (Load Image) not found.")

    # Node 15: Positive Prompt (User input or Default Empty)
    if "15" in workflow:
        workflow["15"]["inputs"]["prompt"] = prompt
        print(f'   Positive Prompt set to: "{prompt}"')
    else:
        sys.exit("[ERROR]: Node 15 (Positive Prompt) not found.")

    # Node 13: Negative Prompt (ALWAYS EMPTY)
    if "13" in workflow:
        workflow["13"]["inputs"]["prompt"] = ""
        print("[INFO]:Negative Prompt forced to empty.")
    else:
        sys.exit("[ERROR]: Node 13 (Negative Prompt) not found.")

    # Randomize Seeds to ensure unique results
    # Node 10 (First KSampler)
    if "10" in workflow:
        seed1 = random.randint(1, 10**14)
        workflow["10"]["inputs"]["seed"] = seed1
        print(f"   Seed (Node 10) set to: {seed1}")

    # Node 33 (Flux KSampler)
    if "33" in workflow:
        seed2 = random.randint(1, 10**14)
        workflow["33"]["inputs"]["seed"] = seed2
        print(f"   Seed (Node 33) set to: {seed2}")

    # 6. Queue Job
    print("[INFO]: Sending job to ComfyUI...", end="")
    response = queue_prompt(workflow)
    prompt_id = response["prompt_id"]
    print(f" [SUCCESS]:  Job ID: {prompt_id}")

    # 7. Monitor Loop
    print("[INFO]: Waiting for generation", end="", flush=True)
    while True:
        history = get_history(prompt_id)
        if prompt_id in history:
            print("\n[DONE] Generation finished.")
            outputs = history[prompt_id].get("outputs", {})

            found_output = False
            for node_id, output_data in outputs.items():
                if "images" in output_data:
                    for img in output_data["images"]:
                        found_output = True
                        print(f"   Server file: {img['filename']}")
                        download_image(img["filename"], img["subfolder"], img["type"])

            if not found_output:
                print("   Warning: Job finished but no output images were found.")
            break
        time.sleep(1)
        print(".", end="", flush=True)
