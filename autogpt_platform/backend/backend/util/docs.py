"""Locate the platform's bundled documentation and its public URLs."""

from functools import cache
from pathlib import Path

# Public docs site. It serves the raw repo docs verbatim, INCLUDING the .md
# extension (https://agpt.co/docs/<repo-relative-path>.md returns the page;
# the extension-less variant 404s for pages outside the site navigation).
DOCS_BASE_URL = "https://agpt.co/docs"


def make_doc_url(path: str) -> str:
    """Public URL for a documentation page (extension kept — see
    ``DOCS_BASE_URL``). Shared by search_docs and get_doc_page so the URL
    shape can't drift between the two tools."""
    return f"{DOCS_BASE_URL}/{path}"


@cache
def get_docs_root(start: Path | None = None) -> Path:
    """Return the ``docs/`` directory shipped with the platform.

    Walks up from *start* (default: this file) until a directory containing
    ``docs/`` is found, so one implementation covers both layouts without
    fragile parent counting: the dev checkout (``<repo>/docs``) and the
    container image (``/app/docs``, see ``COPY docs /app/docs`` in the
    backend Dockerfile). ``docs/platform`` is required as a sentinel so an
    unrelated ``docs`` folder closer to this package (e.g. a future
    ``backend/docs/``) can't shadow the real documentation root.

    *start* exists for testability (point the walk at a tmp tree); results
    are cached per start path.

    Raises FileNotFoundError when no matching directory exists in any
    parent (e.g. a deployment that didn't bundle the docs).
    """
    origin = (start or Path(__file__)).resolve()
    for parent in origin.parents:
        candidate = parent / "docs"
        if (candidate / "platform").is_dir():
            return candidate
    raise FileNotFoundError(
        "docs/ directory not found in any parent of the backend package"
    )
