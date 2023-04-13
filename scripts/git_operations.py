import git

def clone_repository(repo_url, clone_path):
    """Clone a github repository locally"""
    git.Repo.clone_from(repo_url, clone_path)
    result = f"""Cloned {repo_url} to {clone_path}"""

    return result