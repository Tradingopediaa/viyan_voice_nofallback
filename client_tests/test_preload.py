import os
import json
import requests

URL = os.environ["VIYAN_VOICE_URL"]
API_KEY = os.environ.get("RUNPOD_API_KEY")

headers = {"Content-Type": "application/json"}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

r = requests.post(URL, headers=headers, json={"input": {"mode": "preload"}}, timeout=900)
print(r.status_code)
print(json.dumps(r.json(), indent=2))
