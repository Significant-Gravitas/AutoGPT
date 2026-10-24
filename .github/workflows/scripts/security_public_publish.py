"""Update only managed sections of existing, publicly readable releases."""

import argparse
import json
import os
import subprocess
import time

from security_public_api import REPOSITORY, PublicDataError, public_json, public_pages
from security_public_sync import (
    TAG_PATTERN,
    current_snapshot,
    matches,
    public_release_notes,
)


def published_release(release):
    return (
        isinstance(release, dict)
        and type(release.get("id")) is int
        and release["id"] > 0
        and release.get("draft") is False
        and bool(release.get("published_at"))
        and matches(TAG_PATTERN, release.get("tag_name"))
        and isinstance(release.get("body") or "", str)
    )


def patch_release(release_id, body):
    if type(release_id) is not int or release_id <= 0:
        raise PublicDataError("Invalid release write target")
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
        input=json.dumps({"body": body}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if result.returncode:
        raise PublicDataError(
            "Release update failed; rerun after checking public notes"
        )


def update_releases(expected_digest, *, apply=False):
    snapshot = current_snapshot(expected_digest)
    candidates = []
    immutable = 0
    for release in public_pages("releases"):
        if not published_release(release):
            continue
        notes = public_release_notes(
            snapshot, release["tag_name"], release.get("body") or ""
        )
        if notes == (release.get("body") or ""):
            continue
        if release.get("immutable"):
            immutable += 1
            continue
        candidates.append(release)
    # Recheck the public projection after reading release metadata, before any write.
    current_snapshot(expected_digest)
    updated = 0
    for candidate in candidates:
        release, _ = public_json(f"releases/{candidate['id']}")
        if (
            not published_release(release)
            or release["id"] != candidate["id"]
            or release["tag_name"] != candidate["tag_name"]
            or release.get("immutable")
        ):
            raise PublicDataError("Release changed; rerun the workflow")
        body = public_release_notes(
            snapshot, release["tag_name"], release.get("body") or ""
        )
        if body != (release.get("body") or ""):
            if apply:
                patch_release(release["id"], body)
                time.sleep(1)
            updated += 1
    return {"changed_releases": updated, "immutable_releases_skipped": immutable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        if args.apply and (
            os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
            or os.environ.get("GITHUB_REF") != "refs/heads/master"
            or not os.environ.get("GH_TOKEN")
        ):
            raise PublicDataError("Writes require the trusted master workflow")
        print(
            json.dumps(
                update_releases(args.expected_digest, apply=args.apply), sort_keys=True
            )
        )
    except Exception:
        parser.exit(
            1, "Public release sync failed; no raw response or report was logged.\n"
        )


if __name__ == "__main__":
    main()
