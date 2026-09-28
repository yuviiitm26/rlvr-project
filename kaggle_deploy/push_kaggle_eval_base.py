import requests
import json

with open("kaggle_deploy/rlvr_eval.ipynb", "r", encoding="utf-8") as f:
    code_text = f.read()

url = "https://www.kaggle.com/api/v1/kernels/push"
headers = {
    "Authorization": "Bearer KGAT_cd9b331a7678ff3c1044b29e5c307187",
    "Content-Type": "application/json"
}

payload = {
    "slug": "yuvrajgosainiitm/rlvr-project-eval-base",
    "newTitle": "RLVR Project Eval BASE MODEL Phase 6",
    "title": "RLVR Project Eval BASE MODEL Phase 6",
    "text": code_text,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": True,
    "enableInternet": True,
    "datasetDataSources": [],
    "competitionDataSources": [],
    "kernelDataSources": [],  # NO PHASE 5 MOUNTED - this forces fallback to base model
    "modelDataSources": []
}

response = requests.post(url, headers=headers, json=payload)
print(response.status_code)
print(response.text)
