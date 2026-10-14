import os
import json
import requests
import sys

kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
auth = None
headers = {}

if os.path.exists(kaggle_json):
    with open(kaggle_json, "r") as f:
        cfg = json.load(f)
    auth = (cfg["username"], cfg["key"])
else:
    headers["Authorization"] = "Bearer KGAT_cd9b331a7678ff3c1044b29e5c307187"

username = "yuvrajgosainiitm"
slug = "rlvr-project-phase-7"

# 1. Check status
status_url = "https://www.kaggle.com/api/v1/kernels/status"
r_status = requests.get(status_url, params={"userName": username, "kernelSlug": slug}, auth=auth, headers=headers)
print("=== Phase 7 Kernel Status ===")
if r_status.status_code == 200:
    st = r_status.json()
    print(f"Status: {st.get('status')}")
    if st.get("hasFailureMessage"):
        print(f"Failure Message: {st.get('failureMessage')}")
else:
    print(f"Status check failed ({r_status.status_code}): {r_status.text}")

# 2. Check output files if available
output_url = "https://www.kaggle.com/api/v1/kernels/output"
r_out = requests.get(output_url, params={"userName": username, "kernelSlug": slug}, auth=auth, headers=headers)
if r_out.status_code == 200:
    out_data = r_out.json()
    files = out_data.get("files", [])
    print(f"\nAvailable output files ({len(files)}):")
    for f in files:
        print(f" - {f.get('fileName')} ({f.get('totalBytes')} bytes)")
