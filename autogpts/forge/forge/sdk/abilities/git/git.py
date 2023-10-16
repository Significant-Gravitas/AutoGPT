from ..registry import ability

import os
import git
from git import Repo


@ability(
    name="clone_repo",
    description="Clones a GitHub repository.",
    parameters=[
        {"name": "repo_url", "description": "URL of the GitHub repository", "type": "string", "required": True},
        {"name": "dest_path", "description": "Local destination path", "type": "string", "required": True},
    ],
    output_type="string",
)
async def clone_repo(agent, task_id: str, step_id: str, repo_url: str, dest_path: str) -> str:
    try:
        # Retrieve the GitHub token from an environment variable
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            return "GitHub token not found in environment variables."

        # Convert the repo URL to include the token for authentication
        repo_url_with_token = repo_url.replace("https://", f"https://{token}@")

        # Clone the repository
        Repo.clone_from(repo_url_with_token, dest_path)

        return f"Successfully cloned repository to {dest_path}."
    except git.GitCommandError as e:
        return f"Error cloning repository: {str(e)}"



@ability(
    name="create_branch",
    description="Creates a new branch in the repository.",
    parameters=[
        {"name": "branch_name", "description": "Name of the new branch", "type": "string", "required": True},
        {"name": "base_branch", "description": "Base branch for the new branch", "type": "string", "required": False, "default": "main"},
    ],
    output_type="string",
)
async def create_branch(agent, task_id: str, step_id: str, branch_name: str, base_branch: str = 'main') -> str:
    # Again, this would involve Git commands or a library
    return "Successfully created branch."


@ability(
    name="commit_changes",
    description="Commits changes made in the current branch.",
    parameters=[
        {"name": "commit_message", "description": "Message for the commit", "type": "string", "required": True},
    ],
    output_type="string",
)
async def commit_changes(agent, task_id: str, step_id: str, commit_message: str, repo_path: str) -> str:
    try:
        # Load the repository
        repo = Repo(repo_path)
        
        # Check for changes
        if not repo.is_dirty():
            return "No changes to commit."

        # Add all changed files
        repo.git.add(A=True)
        
        # Commit the changes
        repo.git.commit(m=commit_message)

        return f"Successfully committed changes with message: {commit_message}"
    except git.GitCommandError as e:
        return f"Error committing changes: {str(e)}"


@ability(
    name="push_branch",
    description="Pushes the current branch to the specified remote.",
    parameters=[
        {"name": "branch_name", "description": "Name of the branch to push", "type": "string", "required": True},
        {"name": "remote_name", "description": "Name of the remote (default is 'origin')", "type": "string", "required": False, "default": "origin"},
    ],
    output_type="string",
)
async def push_branch(agent, task_id: str, step_id: str, branch_name: str, remote_name: str = 'origin') -> str:
    # Logic for pushing branch goes here
    return "Successfully pushed branch."


@ability(
    name="create_pull_request",
    description="Creates a pull request on GitHub.",
    parameters=[
        {"name": "title", "description": "Title of the pull request", "type": "string", "required": True},
        {"name": "body", "description": "Description/body of the pull request", "type": "string", "required": True},
        {"name": "base_branch", "description": "Branch the changes will be merged into", "type": "string", "required": True},
        {"name": "head_branch", "description": "Branch where your changes are", "type": "string", "required": True},
    ],
    output_type="string",
)
async def create_pull_request(agent, task_id: str, step_id: str, title: str, body: str, base_branch: str, head_branch: str) -> str:
    # Logic for creating a pull request goes here
    return "Successfully created pull request."


@ability(
    name="fetch_issues",
    description="Fetches issues from a GitHub repository.",
    parameters=[
        {"name": "repo_name", "description": "Name of the repository", "type": "string", "required": True},
        {"name": "owner", "description": "Owner of the repository", "type": "string", "required": True},
        {"name": "labels", "description": "Labels to filter issues by", "type": "list", "required": False},
        {"name": "state", "description": "State of issues ('open', 'closed')", "type": "string", "required": False, "default": "open"},
    ],
    output_type="list",
)
async def fetch_issues(agent, task_id: str, step_id: str, repo_name: str, owner: str, labels: list = None, state: str = 'open') -> list:
    # Logic to fetch issues from GitHub goes here
    return []


@ability(
    name="add_comment",
    description="Adds a comment to a specified GitHub issue.",
    parameters=[
        {"name": "repo_name", "description": "Name of the repository", "type": "string", "required": True},
        {"name": "owner", "description": "Owner of the repository", "type": "string", "required": True},
        {"name": "issue_number", "description": "Number of the issue", "type": "int", "required": True},
        {"name": "comment_body", "description": "Content of the comment", "type": "string", "required": True},
    ],
    output_type="string",
)
async def add_comment(agent, task_id: str, step_id: str, repo_name: str, owner: str, issue_number: int, comment_body: str) -> str:
    # Logic to add a comment to a GitHub issue goes here
    return "Successfully added comment to issue."
