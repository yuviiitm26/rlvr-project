import requests
import json

with open("rlvr_eval.ipynb", "r", encoding="utf-8") as f:
    code_text = f.read()

url = "https://www.kaggle.com/api/v1/kernels/push"
headers = {
    "Authorization": "Bearer KGAT_f160f79c240bf3699da2d95699335ff0",
    "Content-Type": "application/json"
}

payload = {
    "slug": "yuvrajgosainiitm/rlvr-project-phase-3-eval",
    "title": "RLVR Project Phase 3 Eval",
    "newTitle": "RLVR Project Phase 3 Eval",
    "text": code_text,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": True,
    "enableGpu": True,
    "enableInternet": True,
    "datasetDataSources": [],
    "competitionDataSources": [],
    "kernelDataSources": ["yuvrajgosainiitm/rlvr-project-phase-2-unsloth"],
    "modelDataSources": []
}

r = requests.post(url, headers=headers, json=payload)
print("Pushing Eval Notebook to Kaggle...")
if r.status_code == 200:
    print("Success!")
    print(json.dumps(r.json(), indent=2))
else:
    print("Failed!", r.status_code)
    print(r.text)
