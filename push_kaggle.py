import requests
import json
import base64
import os

with open("kaggle_deploy/rlvr_train.ipynb", "r", encoding="utf-8") as f:
    code_text = f.read()

url = "https://www.kaggle.com/api/v1/kernels/push"
headers = {
    "Authorization": "Bearer KGAT_cd9b331a7678ff3c1044b29e5c307187",
    "Content-Type": "application/json"
}

payload = {
    "slug": "yuvrajgosainiitm/rlvr-project-phase-5",
    "newTitle": "RLVR Project Phase 5",
    "title": "RLVR Project Phase 5",
    "text": code_text,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": True,
    "enableInternet": True,
    "datasetDataSources": [],
    "competitionDataSources": [],
    "kernelDataSources": [
        "yuvrajgosainiitm/rlvr-project-phase-2-unsloth",
        "yuvrajgosainiitm/rlvr-project-phase-4"
    ],
    "modelDataSources": []
}

response = requests.post(url, headers=headers, json=payload)
print(response.status_code)
print(response.text)
