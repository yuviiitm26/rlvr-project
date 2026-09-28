import os
import requests
import json
import base64

try:
    with open(os.path.join(os.path.dirname(__file__), "secrets.txt"), "r") as f:
        TOKEN = f.read().strip()
except:
    TOKEN = os.environ.get("GITHUB_TOKEN", "REPLACE_ME")
REPO_NAME = "rlvr-project"
username = "yuviiitm26"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

def get_all_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if not f.endswith(".py") and not f.endswith(".ipynb") and not f.endswith(".sh") and not f.endswith(".md"):
                continue
            path = os.path.join(root, f)
            files.append(path)
    return files

def main():
    print("Fetching user info...")
    resp = requests.get("https://api.github.com/user", headers=headers)
    if resp.status_code != 200:
        print("Failed to authenticate:", resp.json())
        return
        
    print(f"Authenticated as: {resp.json()['login']}")

    # Collect files from the organized directories
    files_to_push = []
    # Use absolute paths based on the script location to avoid CWD issues
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ["src", "kaggle_deploy", "tools", "eval_deploy", "docs"]:
        dir_path = os.path.join(repo_root, d)
        if os.path.exists(dir_path):
            files_to_push.extend(get_all_files(dir_path))
            
    # Also grab .py and .md files in the root directory itself
    for f in os.listdir(repo_root):
        path = os.path.join(repo_root, f)
        if os.path.isfile(path) and (f.endswith(".py") or f.endswith(".md")):
            files_to_push.append(path)
            
    files_to_push.append(os.path.join(repo_root, "README.md"))
    
    # Artifacts
    artifacts = [
        r"C:\Users\Yuvra\.gemini\antigravity\brain\f5b7b8c2-15db-46dd-846b-5a7f48ee353f\phase3_results.md",
        r"C:\Users\Yuvra\.gemini\antigravity\brain\f5b7b8c2-15db-46dd-846b-5a7f48ee353f\execution_report.md",
        r"C:\Users\Yuvra\.gemini\antigravity\brain\f5b7b8c2-15db-46dd-846b-5a7f48ee353f\walkthrough.md",
        r"C:\Users\Yuvra\.gemini\antigravity\brain\f5b7b8c2-15db-46dd-846b-5a7f48ee353f\rlvr_architecture.md"
    ]
    files_to_push.extend(artifacts)
    
    ref_url = f"https://api.github.com/repos/{username}/{REPO_NAME}/git/refs/heads/main"
    resp = requests.get(ref_url, headers=headers)
    if resp.status_code != 200:
        print("Failed to get main branch ref:", resp.json())
        return
    commit_sha = resp.json()["object"]["sha"]

    commit_url = f"https://api.github.com/repos/{username}/{REPO_NAME}/git/commits/{commit_sha}"
    base_tree_sha = requests.get(commit_url, headers=headers).json()["tree"]["sha"]

    tree_items = []
    for filename in files_to_push:
        if not os.path.exists(filename):
            continue
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        blob_url = f"https://api.github.com/repos/{username}/{REPO_NAME}/git/blobs"
        blob_resp = requests.post(blob_url, headers=headers, json={"content": content, "encoding": "utf-8"})
        blob_json = blob_resp.json()
        if "sha" not in blob_json:
            print(f"Error uploading {filename}: {blob_json}")
            continue
        blob_sha = blob_json["sha"]
        
        # Clean up path for github repo
        if "brain" in filename:
            repo_path = "artifacts/" + os.path.basename(filename)
        elif filename.startswith(repo_root):
            repo_path = filename[len(repo_root):].lstrip("\\/")
            repo_path = repo_path.replace("\\", "/")
        else:
            repo_path = filename.replace("\\", "/")
            
        tree_items.append({
            "path": repo_path,
            "mode": "100644",
            "type": "blob",
            "sha": blob_sha
        })

    tree_url = f"https://api.github.com/repos/{username}/{REPO_NAME}/git/trees"
    tree_resp = requests.post(tree_url, headers=headers, json={
        "base_tree": base_tree_sha,
        "tree": tree_items
    })
    new_tree_sha = tree_resp.json()["sha"]

    import sys
    if len(sys.argv) > 1:
        commit_message = sys.argv[1]
    else:
        commit_message = "Phase 4 - Automatic update"
    
    commit_resp = requests.post(
        f"https://api.github.com/repos/{username}/{REPO_NAME}/git/commits",
        headers=headers,
        json={
            "message": commit_message,
            "tree": new_tree_sha,
            "parents": [commit_sha]
        }
    )
    new_commit_sha = commit_resp.json()["sha"]

    patch_resp = requests.patch(
        ref_url,
        headers=headers,
        json={"sha": new_commit_sha}
    )
    
    if patch_resp.status_code == 200:
        print(f"Successfully pushed to GitHub! Commit: {new_commit_sha}")
    else:
        print("Failed to update branch reference:", patch_resp.json())

if __name__ == "__main__":
    main()
