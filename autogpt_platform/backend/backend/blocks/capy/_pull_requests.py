"""Find the pull request a Capy agent opened, from what it said in the thread.

Capy's API has no pull-request field on a thread; the agent reports its PR in
its replies. Live replies say "PR #14992 is open against `dev`" as often as
they paste a URL, so a bare number is resolved against the thread's project
when that project covers exactly one repository.
"""

import re

from ._api import CapyClient
from ._types import Message

_PR_URL = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+")
_PR_NUMBER = re.compile(r"\b(?:PR|pull request)\s*#(\d+)", re.IGNORECASE)


async def find_pull_request_url(
    client: CapyClient, project_id: str | None, messages: list[Message]
) -> str:
    """The newest pull request the agent mentioned, as a GitHub URL, or ""."""
    url, number = newest_pull_request_ref(messages)
    if url or not number or not project_id:
        return url
    repo = await _single_repo(client, project_id)
    return f"https://github.com/{repo}/pull/{number}" if repo else ""


def newest_pull_request_ref(messages: list[Message]) -> tuple[str, str]:
    """The newest (url, number) an assistant entry mentions.

    Scans from the newest entry back, because the agent names its PR once and
    may reply several more times after. A full URL wins over a bare number in
    the same entry.
    """
    for message in reversed(messages):
        if message.source != "assistant":
            continue
        if urls := _PR_URL.findall(message.text):
            return urls[-1], ""
        if numbers := _PR_NUMBER.findall(message.text):
            return "", numbers[-1]
    return "", ""


async def _single_repo(client: CapyClient, project_id: str) -> str:
    # Best effort: the PR link is a convenience on top of the wait result,
    # so a failed lookup costs the link, never the thread state.
    try:
        project = await client.get_project(project_id)
    except Exception:
        return ""
    return project.repos[0].repo_full_name if len(project.repos) == 1 else ""
