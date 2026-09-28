import requests, re

url = 'https://www.kaggle.com/code/yuvrajgosainiitm/rlvr-project-phase-5'
headers = {'Authorization': 'Bearer KGAT_cd9b331a7678ff3c1044b29e5c307187'}
response = requests.get(url, headers=headers)
print(response.status_code)
matches = set(re.findall(r'"kernelSessionId":\s*(\d+)', response.text))
print('kernelSessionIds:', matches)

if not matches:
    matches = set(re.findall(r'"id":\s*(\d+)', response.text))
    print('ids:', matches)
