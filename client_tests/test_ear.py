import os
import json
import base64
import requests
import sys

URL = os.environ["VIYAN_VOICE_URL"]
API_KEY = os.environ.get("RUNPOD_API_KEY")
AUDIO = sys.argv[1] if len(sys.argv) > 1 else "test.wav"

headers = {"Content-Type": "application/json"}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

with open(AUDIO, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "input": {
        "mode": "ear",
        "audio_base64": audio_b64,
        "lang": "eng_Latn"
    }
}

r = requests.post(URL, headers=headers, json=payload, timeout=900)
print(r.status_code)
print(json.dumps(r.json(), indent=2, ensure_ascii=False))
