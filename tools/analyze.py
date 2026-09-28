import json

with open(r'C:\Users\Yuvra\.gemini\antigravity\brain\f5b7b8c2-15db-46dd-846b-5a7f48ee353f\.system_generated\steps\2266\output.txt', 'r', encoding='utf-8') as f:
    text = f.read()

data = json.loads(text)
files = data.get("files", [])
for item in files:
    print(item['file_name'])
