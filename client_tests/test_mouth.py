import os
import json
import base64
import requests

URL = os.environ["VIYAN_VOICE_URL"]
API_KEY = os.environ.get("RUNPOD_API_KEY")

headers = {"Content-Type": "application/json"}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

payload = {
    "input": {
        "mode": "mouth",
        "spoken_text": "No, I understand. I am here with you.",
        "language_id": "en",
        "voice_instruct": "male, young adult, low pitch, indian accent",
        "delivery": {
            "speed": 0.95,
            "nonverbal": [
                {"type": "soft_breath", "position": "before_start"}
            ]
        }
    }
}

r = requests.post(URL, headers=headers, json=payload, timeout=900)
print(r.status_code)
data = r.json()
print(json.dumps({k: v for k, v in data.items() if k != "audio_base64"}, indent=2))

if data.get("ok") and data.get("audio_base64"):
    with open("viyan_mouth_test.wav", "wb") as f:
        f.write(base64.b64decode(data["audio_base64"]))
    print("Saved: viyan_mouth_test.wav")
