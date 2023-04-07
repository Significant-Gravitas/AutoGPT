import os
import subprocess

working_directory = "auto_gpt_workspace"

if not os.path.exists(working_directory):
    os.makedirs(working_directory)

def create_github_repo(repo_name, description):
    try:
        github_username = os.getenv("GITHUB_USERNAME")
        github_pat = os.getenv("GITHUB_PAT")

        command = f'curl -u {github_username}:{github_pat} https://api.github.com/user/repos -d \'{{"name":"{repo_name}","description":"{description}"}}\''
        subprocess.run(command, shell=True, check=True, cwd=working_directory, text=True)
        return f"Successfully created a new Github repository named '{repo_name}'."
    except subprocess.CalledProcessError as e:
        return f"An error occurred during the creation of the Github repository: {e}"

def clone_github_repo(repo_url):
    try:
        command = f'git clone {repo_url}'
        subprocess.run(command, shell=True, check=True, cwd=working_directory, text=True)
        return f"Successfully cloned the Github repository from {repo_url}."
    except subprocess.CalledProcessError as e:
        return f"An error occurred while cloning the Github repository: {e}"

def add_file_to_github_repo(repo_name, file_path, commit_message):
    try:
        command = f'cd {repo_name} && git add {file_path} && git commit -m "{commit_message}"'
        subprocess.run(command, shell=True, check=True, cwd=working_directory, text=True)
        return f"Successfully added the file '{file_path}' to the Github repository '{repo_name}'."
    except subprocess.CalledProcessError as e:
        return f"An error occurred while adding the file to the Github repository: {e}"

def push_to_github_repo(repo_name, branch):
    try:
        github_username = os.getenv("GITHUB_USERNAME")
        command = f'cd {repo_name} && git push --set-upstream https://github.com/{github_username}/{repo_name}.git {branch}'
        subprocess.run(command, shell=True, check=True, cwd=working_directory, text=True)
        return f"Successfully pushed to the Github repository '{repo_name}' on branch '{branch}'."
    except subprocess.CalledProcessError as e:
        return f"An error occurred while pushing to the Github repository: {e}"
