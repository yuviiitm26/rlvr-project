import requests
import json
import base64

with open("rlvr_phase2.ipynb", "r", encoding="utf-8") as f:
    code_text = f.read()

url = "https://www.kaggle.com/api/v1/kernels/push"
headers = {
    "Authorization": "Bearer KGAT_f160f79c240bf3699da2d95699335ff0",
    "Content-Type": "application/json"
}

payload = {
    "slug": "yuvrajgosainiitm/rlvr-project-phase-2-unsloth",
    "text": code_text,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": True,
    "enableInternet": True,
    "datasetDataSources": [],
    "competitionDataSources": [],
    "kernelDataSources": [],
    "modelDataSources": []
}

print("Pushing Phase 4 Notebook to Kaggle...")
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    print("Success!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Error {response.status_code}: {response.text}")
