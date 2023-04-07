import os
import subprocess

GITHUB_USERNAME = os.getenv('GITHUB_USERNAME')
GITHUB_PAT = os.getenv('GITHUB_PAT')


def create_github_repo(repo_name):
    """
    Creates a new repository on GitHub with the given name.

    Args:
    repo_name (str): The name of the repository to create.

    Returns:
    str: The URL of the newly created repository.
    """
    command = f"curl -u {GITHUB_USERNAME}:{GITHUB_PAT} https://api.github.com/user/repos -d '{{\"name\":\"{repo_name}\"}}'"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Error creating repository: {stderr.decode('utf-8')}")
    repo_url = f"https://github.com/{GITHUB_USERNAME}/{repo_name}"
    return repo_url


def clone_github_repo(repo_name):
    """
    Clones an existing GitHub repository with the given name.

    Args:
    repo_name (str): The name of the repository to clone.

    Returns:
    None
    """
    repo_url = f"https://github.com/{GITHUB_USERNAME}/{repo_name}.git"
    command = f"git clone {repo_url}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Error cloning repository: {stderr.decode('utf-8')}")


def add_files_to_github_repo(repo_name, files):
    """
    Adds files to an existing GitHub repository with the given name.

    Args:
    repo_name (str): The name of the repository to add files to.
    files (list): A list of file paths to add to the repository.

    Returns:
    None
    """
    for file in files:
        command = f"cd {repo_name} && git add {file}"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Error adding file {file} to repository: {stderr.decode('utf-8')}")


def commit_and_push_to_github_repo(repo_name, message):
    """
    Commits changes to an existing GitHub repository with the given name and pushes them to the remote repository.

    Args:
    repo_name (str): The name of the repository to commit changes to.
    message (str): The commit message.

    Returns:
    None
    """
    command = f"cd {repo_name} && git commit -m \"{message}\" && git push"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Error committing and pushing changes to repository: {stderr.decode('utf-8')}")
