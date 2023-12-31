import json
import os

import requests


def get_pr_labels(repo, pr_number, token):
    """Fetches labels for a given pull request"""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/labels"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)
    return [label["name"] for label in response.json()]


def determine_new_title(labels):
    """Determines the new PR title based on labels"""
    emoji_map = {
        "build": "🏗️",
        "chore": "🔧",
        "lint": "🪄",
        "bug": "🐞",
        "feature": "✨",
        "major feature": "🚀",
        "documentation": "📄",
    }

    emojis = [emoji_map[label] for label in labels if label in emoji_map]
    return " ".join(emojis) + " : " if emojis else ""


def update_pr_title(repo, pr_number, new_title_prefix, token):
    """Updates the PR title with the new prefix"""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {"Authorization": f"token {token}"}
    current_title = requests.get(url, headers=headers).json()["title"]

    # Check if the current title already starts with the correct emojis
    if current_title.startswith(new_title_prefix):
        return "No update needed"

    new_title = new_title_prefix + current_title

    payload = json.dumps({"title": new_title})
    response = requests.patch(url, headers=headers, data=payload)
    return response.status_code


# Example usage
repo_name = os.getenv("GITHUB_REPOSITORY")
pr_number = os.getenv("PULL_REQUEST_NUMBER")
github_token = os.getenv("GITHUB_TOKEN")

labels = get_pr_labels(repo_name, pr_number, github_token)
new_title_prefix = determine_new_title(labels)
status = update_pr_title(repo_name, pr_number, new_title_prefix, github_token)

print(f"Update Status: {status}")
