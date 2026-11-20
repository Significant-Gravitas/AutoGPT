"""The docs that describe the platform's side say what the platform does.

docs/PLATFORM_HOOKS.md was written before the platform side existed. Its
route sketch took the shim's access token as a query parameter. The platform
reads the token from the `Authorization` header only, and someone who
extended the platform from the sketch would have put a long-lived token into
URLs, where proxies and access logs keep it. That doc and
docs/OAUTH_FLOW.md also still said the platform side wasn't merged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = ["README.md", *sorted(f"docs/{path.name}" for path in (ROOT / "docs").glob("*.md"))]

# What a doc said while the platform side was a pull request.
_UNMERGED = re.compile(
    r"not merged|isn't merged|not yet merged|once it merges|(not|isn't) on `?dev"
    r"|on that branch|Spec / Not Implemented",
    re.I,
)


def _text(document: str) -> str:
    return " ".join((ROOT / document).read_text(encoding="utf-8").split())


def _says_a_token_in_the_url_is_refused(text: str) -> bool:
    """Whether `text` says, in the sentence that names `?token=...`, that
    such a connection is refused."""
    about_the_url = [
        sentence for sentence in re.split(r"(?<=[.:])\s", text) if "`?token=...`" in sentence
    ]
    return bool(about_the_url) and all(
        re.search(r"\bis refused\b.*`\?token=\.\.\.`", sentence) for sentence in about_the_url
    )


def test_the_route_in_platform_hooks_takes_the_token_from_the_header() -> None:
    hooks = _text("docs/PLATFORM_HOOKS.md")

    # No parameter of the route is the token.
    assert "Query(" not in hooks
    assert not re.search(r"\btoken: str\b", hooks)
    assert "`Authorization: Bearer <access token>` header" in hooks
    # And it says what happens to a token in the URL: the connection is
    # refused. Naming `?token=...` and the backend's test was all this asked,
    # which a doc that said such a token is taken would have passed.
    assert _says_a_token_in_the_url_is_refused(hooks)
    assert "test_query_string_token_is_not_accepted" in hooks


def test_the_check_for_a_token_in_the_url_reads_what_is_said_of_it() -> None:
    refused = (
        'A connection without that header is refused with "Missing bearer token", also '
        "when the URL carries `?token=...` (`test_query_string_token_is_not_accepted`)."
    )
    taken = (
        "The token is in the `Authorization: Bearer <access token>` header. A token in "
        "the URL, `?token=...`, is accepted too (`test_query_string_token_is_not_accepted`)."
    )

    assert _says_a_token_in_the_url_is_refused(refused)
    assert not _says_a_token_in_the_url_is_refused(taken)
    assert not _says_a_token_in_the_url_is_refused("Nothing here names it.")


def test_platform_hooks_says_the_platform_side_merged() -> None:
    hooks = _text("docs/PLATFORM_HOOKS.md")

    assert "**Status**: the platform side is built" in hooks
    assert "https://github.com/Significant-Gravitas/AutoGPT/pull/13050" in hooks
    # Each section the sketch covers says what was built instead.
    sections = re.split(r"(?m)^## ", (ROOT / "docs/PLATFORM_HOOKS.md").read_text(encoding="utf-8"))
    sketched = [section for section in sections if re.match(r"[1-9]\. ", section)]
    assert len(sketched) == 9
    for section in sketched:
        title = section.splitlines()[0]
        assert "As built" in section or "Superseded by the implementation" in section, title


@pytest.mark.parametrize("document", DOCUMENTS)
def test_no_doc_says_the_platform_side_is_unmerged(document: str) -> None:
    found = _UNMERGED.search(_text(document))

    assert found is None, (document, found.group(0) if found else None)


def test_the_oauth_doc_names_the_migration_that_merged() -> None:
    oauth = _text("docs/OAUTH_FLOW.md")

    assert "20261102120000_add_oauth_application_is_public" in oauth
    assert "20260604120000" not in oauth
