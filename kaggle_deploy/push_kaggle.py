import requests
import json
import os

notebook_path = os.path.join(os.path.dirname(__file__), "rlvr_train.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    code_text = f.read()

kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
auth = None
headers = {"Content-Type": "application/json"}

if os.path.exists(kaggle_json):
    with open(kaggle_json, "r") as f:
        cfg = json.load(f)
    auth = (cfg["username"], cfg["key"])
else:
    headers["Authorization"] = "Bearer KGAT_cd9b331a7678ff3c1044b29e5c307187"

url = "https://www.kaggle.com/api/v1/kernels/push"

payload = {
    "slug": "yuvrajgosainiitm/rlvr-project-phase-7",
    "newTitle": "RLVR Project Phase 7",
    "title": "RLVR Project Phase 7",
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

print("Pushing Phase 7 training kernel to Kaggle...")
if auth:
    response = requests.post(url, auth=auth, headers=headers, json=payload)
else:
    response = requests.post(url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.text)

if response.status_code == 200:
    data = response.json()
    print("\n✅ Successfully launched Kaggle Kernel!")
    print(f"URL: https://www.kaggle.com/code/yuvrajgosainiitm/rlvr-project-phase-7")
