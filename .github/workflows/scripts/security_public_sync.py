"""Generate public credits and release sections from a strict advisory projection."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from security_public_api import PublicDataError, PublicErrorCode
from security_release_notes import (
    BASE_URL,
    END,
    LOGIN_PATTERN,
    START,
    accepted_researchers,
    fetch_advisories,
    fixed_in,
)

SNAPSHOT_PATH = Path(".github/security-advisories.public.json")
CREDIT_START = "<!-- security-credits:start -->"
CREDIT_END = "<!-- security-credits:end -->"
GHSA_PATTERN = r"GHSA-[a-z0-9]{4}(?:-[a-z0-9]{4}){2}"
CVE_PATTERN = r"CVE-\d{4}-\d{4,}"
# Names researchers asked to be credited by, shown beside their handle in SECURITY.md
# and release notes. Add one only at the researcher's request.
RESEARCHER_NAMES = {
    "1024wlsdud": "Dirstibone",
    "DavidCarliez": "David Carliez",
    "Fried-Squid": "Ace",
    "Fushuling": "狗and猫",
    "geckosecurity": "Gecko Security",
    "Greg-Kim": "Minjoon Gregorio Kim",
    "jjjutla": "JJ",
    "MegaManSec": "Joshua Rogers",
    "nkoorty": "Artemiy",
    "pavanchow": "Pavan Nallamothu",
    "programsurf": "Sunwoo Lee",
    "RacerZ-fighting": "RacerZ",
    "sivaadityacoder": "Panuganti Siva Aditya",
    "TrebledJ": "Johnathan",
    "YLChen-007": "Edward-x",
    "yoonsh": "Seunghyun Yoon",
}
TAG_PATTERN = (
    r"(?:autogpt-platform-beta-v|agpt-platform-beta-v|autogpt-v|v)\d+\.\d+\.\d+"
)


def matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def validate_snapshot(snapshot):
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != {"schema_version", "advisories"}
        or type(snapshot["schema_version"]) is not int
        or snapshot["schema_version"] != 1
        or not isinstance(snapshot["advisories"], list)
    ):
        raise PublicDataError(PublicErrorCode.SNAPSHOT_SCHEMA)
    ids = set()
    for item in snapshot["advisories"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"ghsa_id", "cve_id", "researchers", "release_tags"}
            or not matches(GHSA_PATTERN, item["ghsa_id"])
            or item["ghsa_id"] in ids
            or (item["cve_id"] is not None and not matches(CVE_PATTERN, item["cve_id"]))
        ):
            raise PublicDataError(PublicErrorCode.SNAPSHOT_RECORD)
        ids.add(item["ghsa_id"])
        for field, pattern in (
            ("researchers", LOGIN_PATTERN),
            ("release_tags", TAG_PATTERN),
        ):
            values = item[field]
            if (
                not isinstance(values, list)
                or not all(matches(pattern, v) for v in values)
                or len(set(values)) != len(values)
            ):
                raise PublicDataError(PublicErrorCode.SNAPSHOT_FIELD)
    return snapshot


def public_snapshot(advisories):
    records = []
    for advisory in advisories:
        if not isinstance(advisory, dict):
            raise PublicDataError(PublicErrorCode.ADVISORY_RECORD)
        if (
            advisory.get("state") != "published"
            or not advisory.get("published_at")
            or advisory.get("withdrawn_at")
        ):
            continue
        if not matches(GHSA_PATTERN, advisory.get("ghsa_id")):
            raise PublicDataError(PublicErrorCode.ADVISORY_ID)
        tags = set(
            re.findall(
                re.escape(BASE_URL + "/releases/tag/") + "(" + TAG_PATTERN + ")",
                advisory.get("description") or "",
            )
        )
        cve = advisory.get("cve_id")
        try:
            release_tags = sorted(tag for tag in tags if fixed_in(advisory, tag))
        except ValueError:
            raise PublicDataError(PublicErrorCode.RELEASE_MAPPING) from None
        records.append(
            {
                "ghsa_id": advisory["ghsa_id"],
                "cve_id": cve if matches(CVE_PATTERN, cve) else None,
                "researchers": accepted_researchers(advisory),
                "release_tags": release_tags,
            }
        )
    return validate_snapshot(
        {
            "schema_version": 1,
            "advisories": sorted(records, key=lambda item: item["ghsa_id"]),
        }
    )


def current_snapshot(expected_digest=None):
    snapshot = public_snapshot(fetch_advisories())
    if expected_digest is not None and snapshot_digest(snapshot) != expected_digest:
        raise PublicDataError(PublicErrorCode.ADVISORIES_CHANGED)
    return snapshot


def snapshot_text(snapshot):
    return json.dumps(validate_snapshot(snapshot), indent=2, sort_keys=True) + "\n"


def snapshot_digest(snapshot):
    return hashlib.sha256(snapshot_text(snapshot).encode("utf-8")).hexdigest()


def changes(previous, current):
    old, new = (
        {a["ghsa_id"]: a for a in validate_snapshot(s)["advisories"]}
        for s in (previous, current)
    )
    return {
        "added": len(new.keys() - old.keys()),
        "removed": len(old.keys() - new.keys()),
        "updated": sum(old[key] != new[key] for key in old.keys() & new.keys()),
    }


def managed_section(existing, start, end, section, *, append=False):
    if start not in existing and end not in existing and append:
        return existing.rstrip() + ("\n\n" if existing.strip() else "") + section + "\n"
    if existing.count(start) != 1 or existing.count(end) != 1:
        raise PublicDataError(PublicErrorCode.MARKER_PAIR)
    left, right = existing.index(start), existing.index(end)
    if right < left:
        raise PublicDataError(PublicErrorCode.MARKER_ORDER)
    return existing[:left] + section + existing[right + len(end) :]


def researcher_links(logins):
    links = []
    for login in logins:
        link = f"[@{login}](https://github.com/{login})"
        name = RESEARCHER_NAMES.get(login)
        links.append(f"{name} ({link})" if name else link)
    return ", ".join(links)


def acknowledgments(snapshot, existing):
    logins = sorted(
        {
            login
            for a in validate_snapshot(snapshot)["advisories"]
            for login in a["researchers"]
        },
        key=str.casefold,
    )
    section = "\n".join(
        [
            CREDIT_START,
            "### Published advisory credits",
            "",
            "Accepted reporter, finder and analyst credits on published advisories:",
            "",
            *(f"- {researcher_links([login])}" for login in logins),
            "",
            CREDIT_END,
        ]
    )
    return managed_section(existing, CREDIT_START, CREDIT_END, section)


def public_release_notes(snapshot, tag, existing):
    validate_snapshot(snapshot)
    if not matches(TAG_PATTERN, tag):
        raise PublicDataError(PublicErrorCode.RELEASE_TAG)
    selected = [a for a in snapshot["advisories"] if tag in a["release_tags"]]
    if not selected and START not in existing and END not in existing:
        return existing
    lines = [START, "## Security fixes", ""]
    for item in selected:
        ghsa = item["ghsa_id"]
        line = f"- [{ghsa}]({BASE_URL}/security/advisories/{ghsa})"
        if item["cve_id"]:
            line += f" ({item['cve_id']})"
        if item["researchers"]:
            line += ". Thanks to " + researcher_links(item["researchers"])
        lines.append(line + ".")
    if not selected:
        lines.append("No currently published advisories are linked to this release.")
    lines += ["", END]
    return managed_section(existing, START, END, "\n".join(lines), append=True)


def prepare_credits(root, expected_digest):
    for path in (root, root / ".github", root / SNAPSHOT_PATH, root / "SECURITY.md"):
        if path.is_symlink():
            raise PublicDataError(PublicErrorCode.FILE_SYMLINK)
    snapshot = current_snapshot(expected_digest)
    previous = validate_snapshot(
        json.loads((root / SNAPSHOT_PATH).read_text(encoding="utf-8"))
    )
    security = root / "SECURITY.md"
    updated = acknowledgments(snapshot, security.read_text(encoding="utf-8"))
    difference = changes(previous, snapshot)
    security.write_text(updated, encoding="utf-8")
    (root / SNAPSHOT_PATH).write_text(snapshot_text(snapshot), encoding="utf-8")
    return difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("digest")
    credits = subparsers.add_parser("credits")
    credits.add_argument("--root", type=Path, required=True)
    credits.add_argument("--expected-digest", required=True)
    args = parser.parse_args()
    try:
        if args.command == "digest":
            print(snapshot_digest(current_snapshot()))
        else:
            print(
                json.dumps(
                    prepare_credits(args.root, args.expected_digest), sort_keys=True
                )
            )
    except PublicDataError as error:
        parser.exit(1, f"Public security sync failed: {error.code.value}\n")
    except Exception:
        parser.exit(
            1, "Public security sync failed; no raw response or report was logged.\n"
        )


if __name__ == "__main__":
    main()
