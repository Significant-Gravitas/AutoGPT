"""Update only managed sections of existing, publicly readable releases."""

import argparse
import json
import os
import subprocess
import time

from security_public_api import (
    REPOSITORY,
    PublicDataError,
    PublicErrorCode,
    public_json,
    public_pages,
)
from security_public_sync import (
    END,
    START,
    TAG_PATTERN,
    current_snapshot,
    matches,
    public_release_notes,
    unlinked_advisories,
)


def published_release(release):
    return (
        isinstance(release, dict)
        and type(release.get("id")) is int
        and release["id"] > 0
        and release.get("draft") is False
        and bool(release.get("published_at"))
        and matches(TAG_PATTERN, release.get("tag_name"))
        # GitHub returns null for a release with no notes.
        and (release.get("body") is None or isinstance(release["body"], str))
    )


def latest_release_id():
    release, _ = public_json("releases/latest")
    if not published_release(release):
        raise PublicDataError(PublicErrorCode.LATEST_RELEASE)
    return release["id"]


def patch_release(release_id, body, *, latest):
    if type(release_id) is not int or release_id <= 0 or type(latest) is not bool:
        raise PublicDataError(PublicErrorCode.RELEASE_TARGET)
    # GitHub documents make_latest as defaulting to true, so editing an older release
    # without it could take the Latest badge from the current release.
    payload = {"body": body, "make_latest": "true" if latest else "false"}
    result = subprocess.run(
        [
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            "PATCH",
            f"repos/{REPOSITORY}/releases/{release_id}",
            "--silent",
            "--input",
            "-",
        ],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if result.returncode:
        raise PublicDataError(PublicErrorCode.RELEASE_UPDATE)


def update_releases(expected_digest, *, apply=False, preview=False):
    snapshot = current_snapshot(expected_digest)
    candidates = []
    for release in public_pages("releases"):
        if not published_release(release):
            continue
        notes = public_release_notes(
            snapshot, release["tag_name"], release.get("body") or ""
        )
        if notes == (release.get("body") or ""):
            continue
        candidates.append(release)
    # Recheck the public projection after reading release metadata, before any write.
    current_snapshot(expected_digest)
    latest = latest_release_id() if apply and candidates else None
    updated = 0
    sections = []
    for candidate in candidates:
        release, _ = public_json(f"releases/{candidate['id']}")
        if (
            not published_release(release)
            or release["id"] != candidate["id"]
            or release["tag_name"] != candidate["tag_name"]
        ):
            raise PublicDataError(PublicErrorCode.RELEASE_CHANGED)
        body = public_release_notes(
            snapshot, release["tag_name"], release.get("body") or ""
        )
        if body != (release.get("body") or ""):
            if apply:
                patch_release(release["id"], body, latest=release["id"] == latest)
                time.sleep(1)
            updated += 1
            if preview:
                tag = release["tag_name"]
                section = public_release_notes(snapshot, tag, f"{START}\n{END}\n")
                sections.append(
                    f"### [{tag}](https://github.com/{REPOSITORY}/releases/tag/{tag})"
                    f"\n\n```markdown\n{section}```\n"
                )
    if latest is not None and updated and latest_release_id() != latest:
        raise PublicDataError(PublicErrorCode.LATEST_CHANGED)
    result = {"changed_releases": updated}
    if preview:
        unlinked = unlinked_advisories(snapshot)
        result["preview"] = (
            "## Proposed public security sections\n\n"
            f"Releases to update: {updated}\n\n"
            "Published advisories that no release lists: "
            + (
                ", ".join(unlinked)
                + ". Add a link to the release that fixed each one to its description."
                if unlinked
                else "none."
            )
            + "\n\n"
            "Each section below replaces the managed security section or is appended "
            "if none exists. Other release text is preserved. "
            "Existing release text and raw advisory content are not included "
            "in this preview.\n\n" + "\n".join(sections)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-digest", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument(
        "--preview",
        action="store_true",
        help="Show only proposed public security sections",
    )
    args = parser.parse_args()
    try:
        if args.apply and (
            os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
            or os.environ.get("GITHUB_REF") != "refs/heads/master"
            or not os.environ.get("GH_TOKEN")
        ):
            raise PublicDataError(PublicErrorCode.WRITE_CONTEXT)
        result = update_releases(
            args.expected_digest, apply=args.apply, preview=args.preview
        )
        print(result["preview"] if args.preview else json.dumps(result, sort_keys=True))
    except PublicDataError as error:
        parser.exit(1, f"Public release sync failed: {error.code.value}\n")
    except Exception:
        parser.exit(
            1, "Public release sync failed; no raw response or report was logged.\n"
        )


if __name__ == "__main__":
    main()
