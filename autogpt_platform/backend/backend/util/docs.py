"""Locate the platform's bundled documentation directory."""

from functools import cache
from pathlib import Path


@cache
def get_docs_root() -> Path:
    """Return the ``docs/`` directory shipped with the platform.

    Walks up from this file until a directory containing ``docs/`` is found,
    so one implementation covers both layouts without fragile parent
    counting: the dev checkout (``<repo>/docs``) and the container image
    (``/app/docs``, see ``COPY docs /app/docs`` in the backend Dockerfile).
    ``docs/platform`` is required as a sentinel so an unrelated ``docs``
    folder closer to this package (e.g. a future ``backend/docs/``) can't
    shadow the real documentation root.

    Raises FileNotFoundError when no matching directory exists in any
    parent (e.g. a deployment that didn't bundle the docs).
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "docs"
        if (candidate / "platform").is_dir():
            return candidate
    raise FileNotFoundError(
        "docs/ directory not found in any parent of the backend package"
    )
