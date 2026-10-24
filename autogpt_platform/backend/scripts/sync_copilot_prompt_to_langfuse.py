"""Sync the CoPilot base system prompt from this repo to Langfuse.

The text is ``backend.copilot.service._CACHEABLE_SYSTEM_PROMPT`` after its
f-string is evaluated, compared byte for byte with the Langfuse prompt text (no
normalisation, trailing newlines included).

Modes:
  --check               read-only: report which of ``latest`` / ``production``
                        differ from the repo text. Exits 0.
  --publish-latest      create a new text version from the repo text unless
                        ``latest`` already holds it. Langfuse puts ``latest`` on
                        it, so a revert gets a fresh copy of the older text.
  --promote-production  put ``production`` on the version holding the repo text,
                        creating that version first if none does. Creating moves
                        ``latest``, so the text ``latest`` had is then put back
                        on top as a new version. If that write fails, the run
                        fails; ``--publish-latest`` on dev puts ``latest`` back.

Every write mode runs the guards first and makes no writes when Langfuse
already matches, so a rerun is a no-op.

Usage:
    poetry run python -m scripts.sync_copilot_prompt_to_langfuse --check
"""

import argparse
import contextlib
import difflib
import hashlib
import os
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Any

LATEST = "latest"
PRODUCTION = "production"
MIN_PROMPT_LENGTH = 2000
MAX_VERSIONS_SCANNED = 200
DEFAULT_HOST = "https://cloud.langfuse.com"


class GuardError(Exception):
    pass


@dataclass
class RemoteVersion:
    version: int
    text: str
    labels: list[str]


def load_repo_prompt() -> str:
    from backend.copilot.service import _CACHEABLE_SYSTEM_PROMPT

    return _CACHEABLE_SYSTEM_PROMPT


def load_prompt_name() -> str:
    from backend.copilot.config import ChatConfig

    return ChatConfig().langfuse_prompt_name


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_sha() -> str:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check_prompt_text(text: str) -> None:
    if not text.strip():
        raise GuardError("repo prompt is empty")
    if len(text) < MIN_PROMPT_LENGTH:
        raise GuardError(
            f"repo prompt is {len(text)} chars, under the {MIN_PROMPT_LENGTH} minimum"
        )
    if "{{" in text or "}}" in text:
        raise GuardError(
            "repo prompt contains '{{' or '}}', which Langfuse would treat as a "
            "template variable"
        )
    if "{users_information}" in text:
        raise GuardError("repo prompt contains the {users_information} placeholder")


def load_credentials() -> tuple[str, str, str]:
    from backend.util.settings import Settings

    secrets = Settings().secrets
    if not secrets.langfuse_public_key or not secrets.langfuse_secret_key:
        raise GuardError("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")
    return (
        secrets.langfuse_public_key,
        secrets.langfuse_secret_key,
        # An unset CI secret arrives as "", which beats the settings default.
        secrets.langfuse_host or DEFAULT_HOST,
    )


def make_client(public_key: str, secret_key: str, host: str) -> Any:
    from langfuse import Langfuse

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        tracing_enabled=False,
    )


def _to_remote(prompt: Any) -> RemoteVersion:
    if not isinstance(prompt.prompt, str):
        raise GuardError(f"Langfuse version {prompt.version} is not a text prompt")
    return RemoteVersion(prompt.version, prompt.prompt, list(prompt.labels))


def _encoded(name: str) -> str:
    # Same quoting the SDK's own get_prompt applies to the path segment.
    return urllib.parse.quote(name, safe="")


def fetch_label(client: Any, name: str, label: str) -> RemoteVersion | None:
    from langfuse.api.resources.commons.errors.not_found_error import NotFoundError

    try:
        return _to_remote(client.api.prompts.get(_encoded(name), label=label))
    except NotFoundError:
        return None


def find_version_with_text(client: Any, name: str, text: str) -> RemoteVersion | None:
    metas = client.api.prompts.list(name=name, limit=1).data
    if not metas:
        return None
    versions = sorted(metas[0].versions, reverse=True)[:MAX_VERSIONS_SCANNED]
    for version in versions:
        remote = _to_remote(client.api.prompts.get(_encoded(name), version=version))
        if remote.text == text:
            return remote
    return None


def diff_summary(remote: str, local: str, label: str) -> str:
    lines = list(
        difflib.unified_diff(
            remote.splitlines(),
            local.splitlines(),
            fromfile=f"langfuse:{label}",
            tofile="repo",
            lineterm="",
            n=0,
        )
    )
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    head = "\n".join(ln[:200] for ln in lines[:40])
    more = "\n  ..." if len(lines) > 40 else ""
    return f"  +{added} / -{removed} lines\n{head}{more}"


def run_check(client: Any, name: str, text: str) -> list[str]:
    """Print the state of each label and return the labels that are out of date."""
    print(f"Prompt '{name}': repo text {len(text)} chars, sha256 {text_sha256(text)}")
    stale = []
    for label in (LATEST, PRODUCTION):
        remote = fetch_label(client, name, label)
        if remote is None:
            print(f"[{label}] missing in Langfuse")
            stale.append(label)
        elif remote.text == text:
            print(f"[{label}] v{remote.version} matches the repo")
        else:
            print(f"[{label}] v{remote.version} differs from the repo")
            print(diff_summary(remote.text, text, label))
            stale.append(label)
    print("Out of date: " + (", ".join(stale) if stale else "none"))
    return stale


def create_version(
    client: Any, name: str, text: str, reason: str = "Synced from"
) -> RemoteVersion:
    sha, digest = git_sha(), text_sha256(text)
    created = client.create_prompt(
        name=name,
        prompt=text,
        type="text",
        labels=[],
        config={"git_sha": sha, "text_sha256": digest},
        commit_message=f"{reason} AutoGPT {sha} (sha256 {digest})",
    )
    print(f"Created v{created.version}")
    return RemoteVersion(created.version, text, list(created.labels))


def publish_latest(client: Any, name: str, text: str) -> RemoteVersion:
    latest = fetch_label(client, name, LATEST)
    if latest is not None and latest.text == text:
        print(f"[{LATEST}] v{latest.version} already matches the repo, no write")
        return latest
    # Langfuse puts latest only on the newest version, so a revert to text an
    # older version holds still needs a new copy of it.
    return create_version(client, name, text)


def promote_production(client: Any, name: str, text: str) -> RemoteVersion:
    production = fetch_label(client, name, PRODUCTION)
    if production is not None and production.text == text:
        print(
            f"[{PRODUCTION}] v{production.version} already matches the repo, no write"
        )
        return production
    target = find_version_with_text(client, name, text)
    displaced = None
    if target is None:
        displaced = fetch_label(client, name, LATEST)
        target = create_version(client, name, text)
    # Keep the version's other labels; "latest" is reserved and set by Langfuse.
    labels = [lb for lb in target.labels if lb not in (LATEST, PRODUCTION)]
    try:
        client.update_prompt(
            name=name, version=target.version, new_labels=[*labels, PRODUCTION]
        )
    except Exception:
        # Creating the version already took latest off dev's text, and a rerun
        # finds that version and skips the restore, so restore it here.
        if displaced is not None:
            with contextlib.suppress(Exception):
                restore_latest(client, name, displaced, production_moved=False)
        raise
    print(f"[{PRODUCTION}] moved to v{target.version}")
    if displaced is not None:
        restore_latest(client, name, displaced, production_moved=True)
    return target


def restore_latest(
    client: Any, name: str, displaced: RemoteVersion, production_moved: bool
) -> None:
    # Creating the version took latest, which dev serves, off dev's text.
    try:
        restored = create_version(
            client,
            name,
            displaced.text,
            reason=f"Restored v{displaced.version} as {LATEST} after promoting",
        )
    except Exception:
        state = "is done" if production_moved else "was not moved"
        print(
            f"[{LATEST}] could not restore v{displaced.version}'s text; "
            f"{PRODUCTION} {state}. Run --publish-latest on dev to put "
            f"{LATEST} back.",
            file=sys.stderr,
        )
        raise
    print(f"[{LATEST}] restored to v{displaced.version}'s text as v{restored.version}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sync the CoPilot base system prompt to Langfuse."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_const", dest="mode", const="check")
    mode.add_argument(
        "--publish-latest", action="store_const", dest="mode", const="publish-latest"
    )
    mode.add_argument(
        "--promote-production",
        action="store_const",
        dest="mode",
        const="promote-production",
    )
    args = parser.parse_args(argv)
    selected = args.mode or "check"

    # An unset CI secret arrives as LANGFUSE_HOST="", which the SDK takes over
    # its default. Importing backend.copilot.service builds the Langfuse client
    # whose connection every later client with the same key reuses, so drop it
    # before that import.
    if os.environ.get("LANGFUSE_HOST") == "":
        del os.environ["LANGFUSE_HOST"]

    try:
        text = load_repo_prompt()
        check_prompt_text(text)
        client = make_client(*load_credentials())
    except GuardError as exc:
        print(f"Refusing to sync: {exc}", file=sys.stderr)
        return 1

    name = load_prompt_name()
    if selected == "check":
        run_check(client, name, text)
    elif selected == "publish-latest":
        publish_latest(client, name, text)
    else:
        promote_production(client, name, text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
