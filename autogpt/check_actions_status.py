import os
import requests
import sys
import time


# GitHub API endpoint
api_url = os.environ["GITHUB_API_URL"]
repo = os.environ["GITHUB_REPOSITORY"]
sha = os.environ["GITHUB_SHA"]

# GitHub token for authentication
github_token = os.environ["GITHUB_TOKEN"]

# API endpoint for check runs for the specific SHA
endpoint = f"{api_url}/repos/{repo}/commits/{sha}/check-runs"

# Set up headers for authentication
headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/vnd.github.v3+json",
}

# Make the API request
response = requests.get(endpoint, headers=headers)

if response.status_code != 200:
    print(
        f"Error: Unable to fetch check runs data. Status code: {response.status_code}"
    )
    sys.exit(1)

check_runs = response.json()["check_runs"]

# Flag to track if all other check runs have passed
all_others_passed = True

# Current run id
current_run_id = os.environ["GITHUB_RUN_ID"]
# Initialize a flag to check if any runs are still in progress
runs_in_progress = True

while runs_in_progress:
    runs_in_progress = False
    all_others_passed = True

    for run in check_runs:
        if str(run["id"]) != current_run_id:
            status = run["status"]
            conclusion = run["conclusion"]

            if status == "completed":
                if conclusion not in ["success", "skipped", "neutral"]:
                    all_others_passed = False
                    print(
                        f"Check run {run['name']} (ID: {run['id']}) has conclusion: {conclusion}"
                    )
            else:
                runs_in_progress = True
                print(f"Check run {run['name']} (ID: {run['id']}) is still {status}.")
                all_others_passed = False

    if runs_in_progress:
        print(
            "Some check runs are still in progress. Waiting 5 seconds before checking again..."
        )
        time.sleep(5)

        # Fetch updated check runs data
        response = requests.get(endpoint, headers=headers)
        if response.status_code != 200:
            print(
                f"Error: Unable to fetch updated check runs data. Status code: {response.status_code}"
            )
            sys.exit(1)
        check_runs = response.json()["check_runs"]

# After the loop, all_others_passed will have the final result

if all_others_passed:
    print("All other completed check runs have passed. This check passes.")
    sys.exit(0)
else:
    print("Some check runs have failed or have not completed. This check fails.")
    sys.exit(1)
