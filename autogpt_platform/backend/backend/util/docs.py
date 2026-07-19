"""Locate the platform's bundled documentation and its public URLs."""

from functools import cache
from pathlib import Path

# Public docs site. It serves rendered pages at the EXTENSION-LESS repo path
# (https://agpt.co/docs/<repo-relative-path-without-.md>); the .md variant
# returns a soft-404 ("Page Not Found" with HTTP 200), so status-code checks
# alone cannot validate these URLs — grep the body when re-verifying.
DOCS_BASE_URL = "https://agpt.co/docs"


def make_doc_url(path: str) -> str:
    """Public URL for a documentation page (extension stripped, underscores
    canonicalized to hyphens — the site's edge worker rewrites ``_`` to
    ``-``, see ``autogpt_platform/cloudflare_worker.js``). Shared by
    search_docs and get_doc_page so the URL shape can't drift between the
    two tools."""
    clean = path.lstrip("/")
    for ext in (".md", ".mdx"):
        if clean.endswith(ext):
            clean = clean[: -len(ext)]
            break
    return f"{DOCS_BASE_URL}/{clean.replace('_', '-')}"


@cache
def get_docs_root(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: this file) to the bundled docs root,
    or ``None`` when this deployment didn't bundle the docs.

    One implementation covers both layouts without fragile parent counting:
    the dev checkout (``<repo>/docs``) and the container image
    (``/app/docs``, see ``COPY docs /app/docs`` in the backend Dockerfile).
    ``docs/platform`` is required as a sentinel so an unrelated ``docs``
    folder closer to this package (e.g. a future ``backend/docs/``) can't
    shadow the real documentation root.

    *start* exists for testability (point the walk at a tmp tree). Both the
    found path (already resolved — derived from resolved parents) and the
    not-found ``None`` are memoized, so docs-less deployments don't re-walk
    the filesystem on every call.
    """
    origin = (start or Path(__file__)).resolve()
    for parent in origin.parents:
        candidate = parent / "docs"
        if (candidate / "platform").is_dir():
            # Final resolve: robustness if docs/ itself is a symlink, so the
            # containment check in get_doc_page compares real paths.
            return candidate.resolve()
    return None
