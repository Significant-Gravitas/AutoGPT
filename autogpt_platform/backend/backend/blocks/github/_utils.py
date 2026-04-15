def github_repo_path(repo_url: str) -> str:
    """Extract 'owner/repo' from a GitHub repository URL."""
    return repo_url.replace("https://github.com/", "")
