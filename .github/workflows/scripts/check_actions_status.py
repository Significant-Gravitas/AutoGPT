import os
import requests
import sys

# GitHub API endpoint
api_url = os.environ["GITHUB_API_URL"]
repo = os.environ["GITHUB_REPOSITORY"]
run_id = os.environ["GITHUB_RUN_ID"]

# GitHub token for authentication
github_token = os.environ["GITHUB_TOKEN"]

# API endpoint for the current workflow run
endpoint = f"{api_url}/repos/{repo}/actions/runs/{run_id}/jobs"

# Set up headers for authentication
headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/vnd.github.v3+json"
}

# Make the API request
response = requests.get(endpoint, headers=headers)

if response.status_code != 200:
    print(f"Error: Unable to fetch workflow data. Status code: {response.status_code}")
    sys.exit(1)

jobs = response.json()["jobs"]

# Flag to track if all other jobs have passed or are neutral
all_others_passed = True

# Current job name
current_job = os.environ["GITHUB_JOB"]

for job in jobs:
    if job["name"] != current_job:
        status = job["conclusion"]
        if status not in ["success", "neutral", "skipped"]:
            all_others_passed = False
            print(f"Job "{job["name"]}" has status: {status}")

if all_others_passed:
    print("All other jobs have passed or are neutral. This check passes.")
    sys.exit(0)
else:
    print("Some jobs have failed. This check fails.")
    sys.exit(1)