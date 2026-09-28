"""A catalog checkout for readers that run outside a deploy: the expert
style eval and roster-wide tests need the roster without publishing it.

``SKILLS_CATALOG_PATH`` wins. Otherwise the configured ref is resolved to a
commit and downloaded once into the local cache, keyed by that commit, so a
second reader in the same session pays nothing. The resolution itself is
one GitHub API call, made once per process: a roster-wide test file would
otherwise make one per test, which the unauthenticated rate limit CI
runners share does not survive.
"""

import os
from pathlib import Path

from .skill_catalog_release import (
    DEFAULT_CATALOG_REF,
    DEFAULT_CATALOG_REPO,
    CatalogSource,
    fetch_catalog,
    github_headers,
    local_revision,
    resolve_ref,
)

CACHE_DIR_ENV = "SKILLS_CATALOG_CACHE_DIR"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "autogpt" / "skills-catalog"


_resolved: dict[tuple[str, str], CatalogSource] = {}


def catalog_checkout() -> CatalogSource:
    local = os.environ.get("SKILLS_CATALOG_PATH")
    if local:
        root = Path(local)
        return CatalogSource(
            repository=f"file:{root}", revision=local_revision(root), root=root
        )
    repo = os.environ.get("SKILLS_CATALOG_REPO") or DEFAULT_CATALOG_REPO
    ref = os.environ.get("SKILLS_CATALOG_REF") or DEFAULT_CATALOG_REF
    known = _resolved.get((repo, ref))
    if known is not None and known.root.is_dir():
        return known
    source = _download(repo, ref)
    _resolved[(repo, ref)] = source
    return source


def _download(repo: str, ref: str) -> CatalogSource:
    revision = resolve_ref(repo, ref, github_headers())
    cache = Path(os.environ.get(CACHE_DIR_ENV) or _DEFAULT_CACHE_DIR)
    into = cache / repo.replace("/", "__") / revision
    roots = [p for p in into.iterdir() if p.is_dir()] if into.is_dir() else []
    if len(roots) == 1:
        return CatalogSource(repository=repo, revision=revision, root=roots[0])
    into.mkdir(parents=True, exist_ok=True)
    return fetch_catalog(into, repo=repo, ref=revision)
