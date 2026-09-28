import requests, zipfile, io, os

url = "https://www.kaggle.com/api/v1/kernels/output/download_zip/353340049"
headers = {"Authorization": "Bearer KGAT_cd9b331a7678ff3c1044b29e5c307187"}
r = requests.get(url, headers=headers)
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    z.extractall("kaggle_logs_v5")
    print("Extracted files:", z.namelist())
