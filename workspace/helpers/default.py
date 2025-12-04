import json
import urllib.request
import urllib.parse
import urllib.error
import requests  # pip install requests
import time
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
        "inputs": {"samples": ["14", 0], "vae": ["12", 0]},
        "class_type": "VAEDecode",
        "_meta": {"title": "VAE Decode"},
    },
    "3": {
        "inputs": {"shift": 3, "model": ["13", 0]},
        "class_type": "ModelSamplingAuraFlow",
        "_meta": {"title": "ModelSamplingAuraFlow"},
    },
    "4": {
        "inputs": {"filename_prefix": "ComfyUI", "images": ["2", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
    "10": {
        "inputs": {"unet_name": "Qwen-Image-Edit-2509-Q4_0.gguf"},
        "class_type": "UnetLoaderGGUF",
        "_meta": {"title": "Unet Loader (GGUF)"},
    },
    "11": {
        "inputs": {
            "clip_name": "qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image",
            "device": "default",
        },
        "class_type": "CLIPLoader",
        "_meta": {"title": "Load CLIP"},
    },
    "12": {
        "inputs": {"vae_name": "qwen-image/qwen_image_vae.safetensors"},
        "class_type": "VAELoader",
        "_meta": {"title": "Load VAE"},
    },
    "13": {
        "inputs": {
            "lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
            "strength_model": 1,
            "model": ["10", 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LoraLoaderModelOnly"},
    },
    "14": {
        "inputs": {
            "seed": 384211424192731,
            "steps": 5,
            "cfg": 1,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["1", 0],
            "positive": ["17", 0],
            "negative": ["18", 0],
            "latent_image": ["21", 0],
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler"},
    },
    "15": {
        "inputs": {"image": "WhatsApp Image 2025-12-02 at 21.30.27_6ba68f90.jpg"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "17": {
        "inputs": {
            "prompt": "change this scene to a woman rider in full bike riding gear ride the motorcycle. the woman has long, black hair coming out of the helmet she is wearing",
            "clip": ["11", 0],
            "vae": ["12", 0],
            "image": ["15", 0],
        },
        "class_type": "TextEncodeQwenImageEdit",
        "_meta": {"title": "TextEncodeQwenImageEdit"},
    },
    "18": {
        "inputs": {
            "prompt": "",
            "clip": ["11", 0],
            "vae": ["12", 0],
            "image": ["15", 0],
        },
        "class_type": "TextEncodeQwenImageEdit",
        "_meta": {"title": "TextEncodeQwenImageEdit(-ve)"},
    },
    "21": {
        "inputs": {"pixels": ["27", 0], "vae": ["12", 0]},
        "class_type": "VAEEncode",
        "_meta": {"title": "VAE Encode"},
    },
    "27": {
        "inputs": {"upscale_method": "lanczos", "megapixels": 1, "image": ["15", 0]},
        "class_type": "ImageScaleToTotalPixels",
        "_meta": {"title": "ImageScaleToTotalPixels"},
    },
}


# =================================================


def upload_image(image_path):
    """Uploads the input image to ComfyUI. Returns server filename or None on error."""
    url = f"{COMFY_URL}/upload/image"
    print(f"[INFO]:  Uploading {image_path}...", end="")

    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"overwrite": "true"}
            response = requests.post(url, files=files, data=data)
    except Exception as e:
        print(f"\n[ERROR]: exception : {str(e)}")
        return None

    if response is None:
        print("\n[ERROR]: exception : No response from server while uploading.")
        return None

    if response.status_code == 200:
        try:
            server_filename = response.json().get("name")
            print(f" Done! ({server_filename})")
            return server_filename
        except Exception as e:
            print(f"\n[ERROR]: exception : Failed to parse upload response: {str(e)}")
            return None
    else:
        print(
            f"\n[ERROR]: exception : Upload failed: {response.status_code} - {response.text}"
        )
        return None


def queue_prompt(workflow_data):
    """Queues a prompt; returns parsed JSON response or None on error."""
    p = {"prompt": workflow_data}
    data = json.dumps(p).encode("utf-8")
    try:
        req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = "<no body>"
        print(f"\n[ERROR]: exception : SERVER ERROR {e.code} - Message: {body}")
        return None
    except Exception as e:
        print(f"\n[ERROR]: exception : Failed to queue prompt: {str(e)}")
        return None


def get_history(prompt_id):
    """Return JSON history for a prompt id. Raises on error so caller can decide."""
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except Exception as e:
        # propagate up but print first for visibility
        print(f"\n[ERROR]: exception : Failed to get history for {prompt_id}: {str(e)}")
        raise


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


def main(image, prompt, workflow=WORKFLOW):
    workflow = workflow.copy()

    # 2. Upload Input Image
    server_img = upload_image(image)
    if not server_img:
        print(
            f"[ERROR]: exception : Upload failed for image '{image}'. Aborting job setup."
        )
        return None

    print("[INFO]:  Configuring workflow nodes...")

    if "15" in workflow:
        workflow["15"]["inputs"]["image"] = server_img
    else:
        print(
            "[ERROR]: exception : Error: Node 15 (Load Image) not found in workflow JSON."
        )
        return None

    # Node 17: Positive Prompt
    if "17" in workflow:
        workflow["17"]["inputs"]["prompt"] = prompt
        print(f'   Prompt set to: "{prompt}"')
    else:
        print(
            "[ERROR]: exception : Error: Node 17 (Text Encode) not found in workflow JSON."
        )
        return None

    # Node 14: KSampler Seed (Randomize it)
    if "14" in workflow:
        new_seed = random.randint(1, 10**14)
        workflow["14"]["inputs"]["seed"] = new_seed
        print(f"   Seed set to: {new_seed}")

    # 4. Queue Job
    print("[INFO]:  Sending job to ComfyUI...", end="")
    response = queue_prompt(workflow)
    if not response:
        print(
            "\n[ERROR]: exception : Failed to queue prompt; server did not return a valid response."
        )
        return None

    prompt_id = response.get("prompt_id")
    if not prompt_id:
        print(
            "\n[ERROR]: exception : Server response missing 'prompt_id'. Full response:"
        )
        print(json.dumps(response, indent=2))
        return None

    print(f" [SUCCESS]:  Job ID: {prompt_id}")

    # 5. Monitor Loop
    print("[INFO]:  Waiting for generation", end="", flush=True)
    while True:
        try:
            history = get_history(prompt_id)
        except Exception:
            # get_history already printed the error; decide whether to continue polling or abort.
            # Here we abort to avoid spinning forever when server is unreachable.
            print(
                "\n[ERROR]: exception : Aborting monitoring loop due to get_history failure."
            )
            return None

        if prompt_id in history:
            print("\n[DONE] Generation finished.")
            outputs = history[prompt_id].get("outputs", {})

            found_output = False
            for node_id, output_data in outputs.items():
                if "images" in output_data:
                    for img in output_data["images"]:
                        found_output = True
                        print(f"   Server file: {img.get('filename')}")
                        download_image(
                            img.get("filename"), img.get("subfolder"), img.get("type")
                        )

            if not found_output:
                print("   Warning: Job finished but no output images were found.")
            break
        time.sleep(1)
        print(".", end="", flush=True)

    return {"prompt_id": prompt_id, "status": "finished"}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <image_path> <prompt>")
    else:
        image_path = sys.argv[1]
        prompt_text = " ".join(sys.argv[2:])
        main(image_path, prompt_text)
