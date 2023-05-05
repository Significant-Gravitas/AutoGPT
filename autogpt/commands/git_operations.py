"""Git operations for autogpt"""
from git.repo import Repo

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.url_utils.validators import validate_url

CFG = Config()


@command(
    "clone_repository",
    "Clone Repository",
    '"url": "<repository_url>", "clone_path": "<clone_path>"',
    CFG.github_username and CFG.github_api_key,
    "Configure github_username and github_api_key.",
)
@validate_url
def clone_repository(url: str, clone_path: str) -> str:
    """Clone a GitHub repository locally.

    Args:
        url (str): The URL of the repository to clone.
        clone_path (str): The path to clone the repository to.

    Returns:
        str: The result of the clone operation.
    """
    split_url = url.split("//")
    auth_repo_url = f"//{CFG.github_username}:{CFG.github_api_key}@".join(split_url)
    try:
        Repo.clone_from(url=auth_repo_url, to_path=clone_path)
        return f"""Cloned {url} to {clone_path}"""
    except Exception as e:
        return f"Error: {str(e)}"

@command(
    "add",
    "Add files to the staging area",
    '"repo_path": "<repo_path>", "file_path": "<file_path>"',
    CFG.github_username and CFG.github_api_key,
    "Configure github_username and github_api_key.",
)
def add(repo_path: str, file_path: str) -> str:
    """Add a file to the git staging area.

    Args:
        repo_path (str): The path to the repository.
        file_path (str): The path to the file to add.

    Returns:
        str: The result of the add operation.
    """
    try:
        repo = Repo(repo_path)
        repo.index.add([file_path])
        return f"""Added {file_path} to the staging area"""
    except Exception as e:
        return f"Error: {str(e)}"

@command(
    "commit",
    "Commit changes to the repository",
    '"repo_path": "<repo_path>", "message": "<message>"',
    CFG.github_username and CFG.github_api_key,
    "Configure github_username and github_api_key.",
)
def commit(repo_path: str, message: str) -> str:
    """Commit changes to the git repository.

    Args:
        repo_path (str): The path to the repository.
        message (str): The commit message.

    Returns:
        str: The result of the commit operation.
    """
    try:
        repo = Repo(repo_path)
        repo.index.commit(message)
        return f"""Committed changes with message: {message}"""
    except Exception as e:
        return f"Error: {str(e)}"

@command(
    "push",
    "Push changes to a remote repository",
    '"repo_path": "<repo_path>", "remote_name": "<remote_name>", "branch_name": "<branch_name>"',
    CFG.github_username and CFG.github_api_key,
    "Configure github_username and github_api_key.",
)
def push(repo_path: str, remote_name: str, branch_name: str) -> str:
    """Push changes to a remote git repository.

    Args:
        repo_path (str): The path to the repository.
        remote_name (str): The name of the remote repository.
        branch_name (str): The name of the branch to push to.

    Returns:
        str: The result of the push operation.
    """
    try:
        repo = Repo(repo_path)
        origin = repo.remote(name=remote_name)
        origin.push(refspec=f"HEAD:{branch_name}")
        return f"""Pushed changes to {remote_name}/{branch_name}"""
    except Exception as e:
        return f"Error: {str(e)}"

@command(
    "pull",
    "Pull changes from a remote repository",
    '"repo_path": "<repo_path>", "remote_name": "<remote_name>", "branch_name": "<branch_name>"',
    CFG.github_username and CFG.github_api_key,
    "Configure github_username and github_api_key.",
)
@command(
    "pull",
    "Pull changes from a remote repository",
    '"repo_path": "<repo_path>", "remote_name": "<remote_name>", "branch_name": "<branch_name>"',
    CFG.github_username and CFG.github_api_key,
    "Configure github_username and github_api_key.",
)
def pull(repo_path: str, remote_name: str, branch_name: str) -> str:
    """Pull changes from a remote git repository.

    Args:
        repo_path (str): The path to the repository.
        remote_name (str): The name of the remote repository.
        branch_name (str): The name of the branch to pull from.

    Returns:
        str: The result of the pull operation.
    """
    try:
        repo = Repo(repo_path)
        origin = repo.remote(name=remote_name)
        origin.pull(branch_name)
        return f"""Pulled changes from {remote_name}/{branch_name}"""
    except Exception as e:
        return f"Error: {str(e)}"
    
@command(
    "init_repository",
    "Initialize a new git repository",
    '"repo_path": "<repo_path>"',
    CFG.github_username and CFG.github_api_key,
    "Configure github_username and github_api_key.",
)
def init_repository(repo_path: str) -> str:
    """Initialize a new git repository.

    Args:
        repo_path (str): The path to initialize the repository at.

    Returns:
        str: The result of the init operation.
    """
    try:
        Repo.init(path=repo_path)
        return f"""Initialized a new git repository at {repo_path}"""
    except Exception as e:
        return f"Error: {str(e)}"


