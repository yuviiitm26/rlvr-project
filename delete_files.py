import requests

TOKEN = "<GITHUB_TOKEN>"
REPO = "yuviiitm26/rlvr-project"
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

files_to_delete = [
    "artifacts/regex_masking_explanation.md",
    "artifacts/multi_turn_implementation_plan.md"
]

for file_path in files_to_delete:
    # Get the file's current SHA
    url = f"https://api.github.com/repos/{REPO}/contents/{file_path}"
    resp = requests.get(url, headers=headers)
    
    if resp.status_code == 200:
        file_sha = resp.json()["sha"]
        
        # Delete the file
        delete_resp = requests.delete(
            url,
            headers=headers,
            json={
                "message": f"Remove deprecated artifact: {file_path}",
                "sha": file_sha
            }
        )
        if delete_resp.status_code == 200:
            print(f"Successfully deleted {file_path} from GitHub.")
        else:
            print(f"Failed to delete {file_path}: {delete_resp.json()}")
    else:
        print(f"File {file_path} not found on GitHub. ({resp.status_code})")
