import requests
import json
import time

BASE = "http://localhost:8000"

# Correct absolute image paths on the server machine
CONTENT = "/workspace/AIP/workspace/outputs/images/main.png"
STYLE = "/workspace/AIP/workspace/outputs/images/reference.png"

timings = {}

# Will be filled dynamically from AI suggestions
IMAGE_ANALYSIS = None


def timed(name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    timings[name] = round(time.time() - start, 3)
    return result


def pretty(resp):
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print("[ERROR] Could not parse JSON response:", e)
        print(resp.status_code, resp.text)


# ---------------------------
# API CALLS
# ---------------------------


def call_health():
    r = requests.get(f"{BASE}/health")
    pretty(r)


def call_status():
    r = requests.get(f"{BASE}/status")
    pretty(r)


def style_transfer_text():
    print("\n--- /style-transfer/text ---")
    r = requests.post(
        f"{BASE}/style-transfer/text",
        json={
            "content": CONTENT,
            "style_text": "soft pastel art style with gentle tones",
            "prompt": "apply a smooth dreamy pastel look",
            "steps": 40,
            "style_steps": 20,
        },
    )
    pretty(r)


def style_transfer_ref():
    print("\n--- /style-transfer/ref ---")
    r = requests.post(
        f"{BASE}/style-transfer/ref",
        json={
            "content": CONTENT,
            "style": STYLE,
            "prompt": "match lighting and texture from the reference",
            "steps": 50,
        },
    )
    pretty(r)


def color_grading():
    print("\n--- /color-grading ---")
    r = requests.post(
        f"{BASE}/color-grading",
        json={
            "image": CONTENT,
            "prompt": "cinematic teal shadows and warm highlights",
            "mode": "both",
        },
    )
    pretty(r)


def ai_suggestions():
    """
    Capture AI suggestions AND store the 'analysis' output globally
    so classify_quick() and classify_full() can reuse it.
    """
    global IMAGE_ANALYSIS

    print("\n--- /ai-suggestions ---")
    r = requests.post(f"{BASE}/ai-suggestions", json={"image": CONTENT})

    pretty(r)

    try:
        data = r.json()

        # We assume your server returns something like:
        # { "analysis": { ...slots... }, ... }
        IMAGE_ANALYSIS = data

    except Exception as e:
        print("[ERROR] Could not parse AI suggestions JSON. e", str(e))

    return r


def classify_quick():
    print("\n--- /classify (quick) ---")
    body = {
        "prompt": "increase sharpness and reduce noise",
        "quick": True,
        "image_description": "auto-generated from ai-suggestions",
        "image_analysis": IMAGE_ANALYSIS,
    }

    r = requests.post(f"{BASE}/classify", json=body)
    pretty(r)


def classify_full():
    print("\n--- /classify (full) ---")
    body = {
        "prompt": "convert to monochrome film look",
        "image_description": "auto-generated from ai-suggestions",
        "image_analysis": IMAGE_ANALYSIS,
    }

    r = requests.post(f"{BASE}/classify", json=body)
    pretty(r)


def sam_segment():
    print("\n--- /sam/segment ---")
    r = requests.post(
        f"{BASE}/sam/segment",
        json={
            "image": CONTENT,
            "x": 150,
            "y": 200,
            "output_dir": "outputs/segmentation",
        },
        timeout=120,
    )
    pretty(r)


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    timed("health", call_health)
    timed("status", call_status)
    timed("style_transfer_text", style_transfer_text)
    timed("style_transfer_ref", style_transfer_ref)
    timed("color_grading", color_grading)

    # This will fill IMAGE_ANALYSIS
    timed("ai_suggestions", ai_suggestions)

    # Now classifier will use real AI analysis
    timed("classify_quick", classify_quick)
    timed("classify_full", classify_full)

    timed("sam_segment", sam_segment)

    print("\n==================== TIMING SUMMARY ====================")
    for k, v in timings.items():
        print(f"{k:25s}: {v:6.3f} sec")
    print("========================================================")

