import re
from urllib.parse import urlparse

_GITHUB_HOSTS = ("github.com", "www.github.com")
_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def normalize_repo_path(repo: str) -> str:
    """
    Normalise a repository reference to 'owner/repo'. Accepts both the bare
    'owner/repo' form and a full github.com URL, and rejects anything else.

    The input is validated, so the result is always safe to interpolate into
    an API path.
    """
    value = repo.strip()
    if "://" in value or value.lower().startswith(_GITHUB_HOSTS):
        parsed = urlparse(value if "://" in value else f"https://{value}")
        if (parsed.hostname or "").lower() not in _GITHUB_HOSTS:
            raise ValueError(
                f"Not a github.com repository URL: {repo!r}. "
                "Expected '{owner}/{repo}' or 'https://github.com/{owner}/{repo}'."
            )
        value = parsed.path

    parts = [part for part in value.strip("/").split("/") if part]
    if len(parts) == 2:
        owner, name = parts[0], parts[1].removesuffix(".git")
        if _OWNER_RE.match(owner) and _REPO_NAME_RE.match(name) and name.strip("."):
            return f"{owner}/{name}"

    raise ValueError(
        f"Invalid repository {repo!r}. "
        "Expected '{owner}/{repo}' or 'https://github.com/{owner}/{repo}'."
    )


def canonical_repo_url(repo_url: str) -> str:
    """
    Normalise a repository reference to 'https://github.com/{owner}/{repo}',
    so callers can append API sub-paths to it. Raises ValueError like
    `normalize_repo_path`.
    """
    return f"https://github.com/{normalize_repo_path(repo_url)}"
