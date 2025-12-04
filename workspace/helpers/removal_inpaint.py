import json
import urllib.request
import urllib.parse
import urllib.error
import requests  # pip install requests
import time
import os


# Polling configuration
MAX_POLL_TIME_SECONDS = 300  # 5 minutes max wait
POLL_INTERVAL_SECONDS = 1

WORKFLOW = {
    "1": {
        "inputs": {"image": "WhatsApp Image 2025-12-02 at 21.30.27_6ba68f90.jpg"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "2": {
        "inputs": {"image": "WhatsApp Image 2025-12-02 at 21.56.20_90168677.jpg"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "3": {
        "inputs": {"channel": "red", "image": ["2", 0]},
        "class_type": "ImageToMask",
        "_meta": {"title": "Convert Image to Mask"},
    },
    "4": {
        "inputs": {
            "lama_model": "lama",
            "device": "cpu",
            "invert_mask": False,
            "mask_grow": 25,
            "mask_blur": 8,
            "image": ["1", 0],
            "mask": ["3", 0],
        },
        "class_type": "LayerUtility: LaMa",
        "_meta": {"title": "LayerUtility: LaMa(Advance)"},
    },
    "5": {
        "inputs": {"filename_prefix": "LaMa_Result", "images": ["4", 0]},
        "class_type": "SaveImage",
        "_meta": {"title": "Save Image"},
    },
}

OUTPUT_FOLDER = "/workspace/comfy_img_remix/comfycode/outputs"

COMFY_URL = "http://0.0.0.0:8001"


# ===========================================================
# UPLOAD IMAGE
# ===========================================================
def upload_image(image_path, type_desc):
    """Uploads an image to ComfyUI and returns the server filename."""

    if not os.path.exists(image_path):
        print(f"[ERROR]     : {type_desc} file not found: {image_path}")
        return None

    url = f"{COMFY_URL}/upload/image"
    print(
        f"[INFO]      : Uploading {type_desc}: {os.path.basename(image_path)}...",
        end="",
    )

    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"overwrite": "true"}
            response = requests.post(url, files=files, data=data)
    except requests.exceptions.ConnectionError:
        print(
            f"\n[ERROR]     : Cannot connect to ComfyUI at {COMFY_URL}. Is it running?"
        )
        return None
    except Exception as e:
        print(f"\n[ERROR]     : Upload exception: {str(e)}")
        return None

    if response.status_code == 200:
        try:
            server_filename = response.json().get("name")
        except Exception:
            print("\n[ERROR]     : Failed to parse server response")
            return None

        print(f" [SUCCESS]  : {server_filename}")
        return server_filename

    print(f"\n[ERROR]     : Upload failed: {response.status_code} - {response.text}")
    return None


# ===========================================================
# QUEUE PROMPT
# ===========================================================
def queue_prompt(workflow_data):
    p = {"prompt": workflow_data}
    data = json.dumps(p).encode("utf-8")

    try:
        req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data)
        raw = urllib.request.urlopen(req).read()
        return json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            msg = e.read().decode("utf-8")
        except Exception:
            msg = "<no body>"
        print(f"\n[ERROR]     : Server returned HTTP {e.code}")
        print(f"[ERROR]     : {msg}")
        return None
    except Exception as e:
        print(f"\n[ERROR]     : Failed to queue prompt: {str(e)}")
        return None


def get_history(prompt_id):
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"[ERROR]     : Failed to read history for {prompt_id}: {str(e)}")
        return None


def download_image(filename, subfolder, folder_type):
    """Downloads generated image from ComfyUI to OUTPUT_FOLDER. Returns path or None."""

    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        with urllib.request.urlopen(f"{COMFY_URL}/view?{url_values}") as response:
            img_data = response.read()
            path = os.path.join(OUTPUT_FOLDER, filename)
            with open(path, "wb") as f:
                f.write(img_data)

        print(f"   [SUCCESS]  : Saved to {path}")
        return path

    except Exception as e:
        print(f"   [ERROR]     : Could not download image: {e}")
        return None


def delete_comfy_input_image(server_filename):
    """Deletes an uploaded image from ComfyUI's input folder."""
    if not server_filename:
        return
    try:
        url = f"{COMFY_URL}/api/delete"
        data = json.dumps(
            {"delete": [{"filename": server_filename, "type": "input"}]}
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req)
        print(f"   [CLEANUP] Deleted input image: {server_filename}")
    except Exception as e:
        print(f"   [CLEANUP] Could not delete {server_filename}: {e}")


def removal_inpaint(input_image, mask_image, workflow=WORKFLOW):
    """
    Run the removal/inpaint workflow using LaMa model.

    Args:
        input_image: Path to input image
        mask_image: Path to mask image (areas to inpaint)
        workflow: ComfyUI workflow dict (optional)

    Returns:
        dict with prompt_id, status, and output_images on success, None on failure
    """
    workflow = workflow.copy()
    server_input = None
    server_mask = None
    output_images = []

    try:
        # 1. Upload Input Images
        server_input = upload_image(input_image, "Input Image")
        server_mask = upload_image(mask_image, "Mask Image")

        if not server_input or not server_mask:
            print("[ERROR]     : Upload failed. Aborting workflow.")
            return None

        print("[INFO]      : Configuring workflow nodes...")

        # Node 1
        if "1" in workflow:
            workflow["1"]["inputs"]["image"] = server_input
            print(f"[INFO]      : Node 1 (Input) set to {server_input}")
        else:
            print("[ERROR]     : Node 1 (Load Image) missing")
            return None

        # Node 2
        if "2" in workflow:
            workflow["2"]["inputs"]["image"] = server_mask
            print(f"[INFO]      : Node 2 (Mask) set to {server_mask}")
        else:
            print("[ERROR]     : Node 2 (Load Mask) missing")
            return None

        # 2. Queue Job
        print("[INFO]      : Sending job to ComfyUI...", end="")
        response = queue_prompt(workflow)

        if not response:
            print("\n[ERROR]     : No valid response from server.")
            return None

        prompt_id = response.get("prompt_id")
        if not prompt_id:
            print("\n[ERROR]     : Server response missing prompt_id")
            return None

        print(f" [SUCCESS]  : Job ID = {prompt_id}")

        # 3. Monitor Loop with timeout
        print("[INFO]      : Waiting for generation", end="", flush=True)
        start_time = time.time()
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            elapsed = time.time() - start_time
            if elapsed > MAX_POLL_TIME_SECONDS:
                print(f"\n[ERROR]: Timeout after {MAX_POLL_TIME_SECONDS}s.")
                return None

            history = get_history(prompt_id)

            if history is None:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(
                        f"\n[ERROR]: Aborting after {max_consecutive_errors} "
                        "consecutive history fetch failures."
                    )
                    return None
                print("x", end="", flush=True)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            consecutive_errors = 0  # Reset on success

            if prompt_id in history:
                print("\n[SUCCESS]  : Generation completed.")

                outputs = history[prompt_id].get("outputs", {})

                for node_id, output_data in outputs.items():
                    if "images" in output_data:
                        for img in output_data["images"]:
                            print(f"[INFO]      : Server file = {img['filename']}")
                            local_path = download_image(
                                img["filename"], img["subfolder"], img["type"]
                            )
                            if local_path:
                                output_images.append(local_path)

                if not output_images:
                    print("[INFO]      : No images found in output.")
                break

            print(".", end="", flush=True)
            time.sleep(POLL_INTERVAL_SECONDS)

        return {
            "prompt_id": prompt_id,
            "status": "finished",
            "output_images": output_images,
        }

    finally:
        # Cleanup: delete uploaded input images from ComfyUI
        if server_input:
            delete_comfy_input_image(server_input)
        if server_mask:
            delete_comfy_input_image(server_mask)
