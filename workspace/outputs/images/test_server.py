#!/usr/bin/env python3
"""
test_api_uploads.py
Client tests for the Flask AI Image Processing Server using file uploads.

All endpoints are timed and printed in a summary at the end.
"""

import requests
import json
import time
import mimetypes
from pathlib import Path

BASE = "http://localhost:8000"
TIMEOUT = 120

CONTENT_IMG = Path("./final_metric.png")
STYLE_IMG = Path("./style_transfer_text_generated_style_1764690427.png")

timings = {}   # <---- store all timings here


def _guess_mime(p: Path):
    mime, _ = mimetypes.guess_type(str(p))
    return mime or "application/octet-stream"


def timed_call(name, func, *args, **kwargs):
    """Times any function and stores duration."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    timings[name] = round(end - start, 3)
    return result


def safe_post_files(url: str, files: dict, data: dict = None, json_payload: dict = None):
    try:
        if json_payload:
            r = requests.post(url, json=json_payload, timeout=TIMEOUT)
        else:
            r = requests.post(url, files=files, data=data or {}, timeout=TIMEOUT)

        try:
            print(json.dumps(r.json(), indent=2))
        except:
            print(r.status_code, r.text)

        return r
    except Exception as e:
        print(f"[REQUEST FAILED] {e}")
        return None


# ---------------------------------------------------------
# API CALL FUNCTIONS
# ---------------------------------------------------------

def style_transfer_text():
    url = f"{BASE}/style-transfer/text"
    mime = _guess_mime(CONTENT_IMG)

    with CONTENT_IMG.open("rb") as f:
        files = {"content": (CONTENT_IMG.name, f, mime)}
        data = {
            "style_text": "soft pastel painting",
            "prompt": "apply gentle pastel tones",
            "steps": "40",
            "style_steps": "20",
        }
        print("\n--- /style-transfer/text ---")
        return safe_post_files(url, files, data=data)


def style_transfer_ref():
    url = f"{BASE}/style-transfer/ref"
    mime_c = _guess_mime(CONTENT_IMG)
    mime_s = _guess_mime(STYLE_IMG)

    with CONTENT_IMG.open("rb") as fc, STYLE_IMG.open("rb") as fs:
        files = {
            "content": (CONTENT_IMG.name, fc, mime_c),
            "style": (STYLE_IMG.name, fs, mime_s),
        }
        data = {"prompt": "match lighting and texture", "steps": "50"}
        print("\n--- /style-transfer/ref ---")
        return safe_post_files(url, files, data=data)


def color_grading():
    url = f"{BASE}/color-grading"
    mime = _guess_mime(CONTENT_IMG)

    with CONTENT_IMG.open("rb") as f:
        files = {"image": (CONTENT_IMG.name, f, mime)}
        data = {
            "prompt": "cinematic teal-orange contrast boost",
            "mode": "both",
        }
        print("\n--- /color-grading ---")
        return safe_post_files(url, files, data=data)


def ai_suggestions():
    url = f"{BASE}/ai-suggestions"
    mime = _guess_mime(CONTENT_IMG)

    with CONTENT_IMG.open("rb") as f:
        files = {"image": (CONTENT_IMG.name, f, mime)}
        print("\n--- /ai-suggestions ---")
        return safe_post_files(url, files)


def classify_quick():
    url = f"{BASE}/classify"
    payload = {
        "prompt": "increase sharpness and reduce noise",
        "quick": True,
    }
    print("\n--- /classify (quick) ---")
    return safe_post_files(url, None, json_payload=payload)


def classify_full():
    url = f"{BASE}/classify"
    payload = {
        "prompt": "convert this to a monochrome film aesthetic"
    }
    print("\n--- /classify (full) ---")
    return safe_post_files(url, None, json_payload=payload)


def health():
    print("\n--- /health ---")
    r = requests.get(f"{BASE}/health")
    print(json.dumps(r.json(), indent=2))


def status():
    print("\n--- /status ---")
    r = requests.get(f"{BASE}/status")
    print(json.dumps(r.json(), indent=2))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    # Run all and time them
    timed_call("health", health)
    timed_call("status", status)
    timed_call("style_transfer_text", style_transfer_text)
    timed_call("style_transfer_ref", style_transfer_ref)
    timed_call("color_grading", color_grading)
    timed_call("ai_suggestions", ai_suggestions)
    timed_call("classify_quick", classify_quick)
    timed_call("classify_full", classify_full)

    # -----------------------------------------------------
    # PRINT SUMMARY
    # -----------------------------------------------------
    print("\n==================== TIMING SUMMARY ====================")
    for k, v in timings.items():
        print(f"{k:25s}: {v:6.3f} sec")
    print("========================================================\n")

