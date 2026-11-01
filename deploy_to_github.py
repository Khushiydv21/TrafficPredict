"""
Automated GitHub Uploader for TrafficPredict
Author: Khushi Yadav

This script uses the official GitHub REST API to automatically create
the repository and upload all project files without requiring git.exe.

Usage:
1. Generate a GitHub Token at: https://github.com/settings/tokens (classic token with 'repo' scope)
2. Run:
   python deploy_to_github.py
"""

import os
import sys
import base64
import requests
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_NAME = "TrafficPredict"
GITHUB_USERNAME = "Khushiydv21"

EXCLUDE_DIRS = {"__pycache__", ".mpl_cache", ".git", "venv", ".streamlit"}
EXCLUDE_FILES = {".DS_Store"}


def create_or_get_repo(token: str):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    # Check if repo exists
    check_url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}"
    r = requests.get(check_url, headers=headers)
    if r.status_code == 200:
        print(f"  * Repository '{GITHUB_USERNAME}/{REPO_NAME}' already exists on GitHub.")
        return True
    
    # Create repo
    print(f"  * Creating repository '{REPO_NAME}' on GitHub...")
    create_url = "https://api.github.com/user/repos"
    payload = {
        "name": REPO_NAME,
        "description": "Predicting Website Traffic Using Machine Learning and Time Series Forecasting (Streamlit Dashboard)",
        "private": False,
        "auto_init": False
    }
    r = requests.post(create_url, json=payload, headers=headers)
    if r.status_code in [201, 200]:
        print(f"  * Successfully created repository: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
        return True
    else:
        print(f"  ! Error creating repo: {r.status_code} - {r.text}")
        return False


def upload_file(file_path: Path, token: str):
    rel_path = file_path.relative_to(PROJECT_DIR).as_posix()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{rel_path}"

    with open(file_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Check if file exists to get sha
    sha = None
    r_check = requests.get(url, headers=headers)
    if r_check.status_code == 200:
        sha = r_check.json().get("sha")

    payload = {
        "message": f"Upload {rel_path}",
        "content": content_b64,
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, json=payload, headers=headers)
    if r.status_code in [200, 201]:
        print(f"    ✔ Uploaded: {rel_path}")
    else:
        print(f"    ✖ Failed {rel_path}: {r.status_code}")


def main():
    print("=" * 70)
    print("  TRAFFICPREDICT - AUTOMATED GITHUB DEPLOYMENT SCRIPT")
    print("=" * 70)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("\nPlease enter your GitHub Personal Access Token (classic token with 'repo' scope):")
        print("Generate one here: https://github.com/settings/tokens")
        token = input("GitHub Token: ").strip()

    if not token:
        print("No token provided. Aborting.")
        return

    success = create_or_get_repo(token)
    if not success:
        return

    print("\nUploading project files to GitHub...")
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file in EXCLUDE_FILES or file.endswith(".pyc"):
                continue
            full_path = Path(root) / file
            upload_file(full_path, token)

    print("\n" + "=" * 70)
    print(f"  SUCCESS! Your code is now live on GitHub:")
    print(f"  👉 https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
    print("=" * 70)
    print("\nNext step: Deploy to Streamlit Cloud in 3 clicks:")
    print("1. Go to: https://share.streamlit.io")
    print(f"2. Select repo '{GITHUB_USERNAME}/{REPO_NAME}' and main file 'app.py'")
    print("3. Click 'Deploy'!")


if __name__ == "__main__":
    main()
