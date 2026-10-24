"""Generate release-note links from published repository security advisories."""

import argparse
import json
import re
import subprocess
from pathlib import Path

REPOSITORY = "Significant-Gravitas/AutoGPT"
BASE_URL = f"https://github.com/{REPOSITORY}"
START = "<!-- security-advisories:start -->"
END = "<!-- security-advisories:end -->"
RESEARCH_ROLES = {"reporter", "finder", "analyst"}


def published_advisories(advisories):
    return [
        advisory
        for advisory in advisories
        if advisory.get("state") == "published"
        and advisory.get("published_at")
        and not advisory.get("withdrawn_at")
        and re.fullmatch(r"GHSA-[a-z0-9]{4}(?:-[a-z0-9]{4}){2}", advisory["ghsa_id"])
    ]


def accepted_researchers(advisory):
    return sorted(
        {
            credit["user"]["login"]
            for credit in advisory.get("credits_detailed") or []
            if credit.get("state") == "accepted"
            and credit.get("type") in RESEARCH_ROLES
            and re.fullmatch(r"[A-Za-z0-9-]+", credit["user"]["login"])
        },
        key=str.casefold,
    )


def version_number(value):
    match = re.fullmatch(
        r"(?:>=\s*)?(?:autogpt-platform-beta-v|v)?(\d+\.\d+\.\d+)",
        (value or "").strip(),
    )
    return match[1] if match else None


def fixed_in(advisory, tag):
    version = version_number(tag)
    if not version:
        raise ValueError(
            "Use a stable release tag, such as autogpt-platform-beta-v0.9.0"
        )
    release_url = f"{BASE_URL}/releases/tag/{tag}"
    has_reference = re.search(
        re.escape(release_url) + r"(?=[\s)#?<>]|$)", advisory.get("description") or ""
    )
    return bool(has_reference) and any(
        version_number(vulnerability.get("patched_versions")) == version
        for vulnerability in advisory.get("vulnerabilities") or []
    )


def inline_text(value):
    value = " ".join(value.split())
    return re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", value)


def release_notes(advisories, tag, existing=""):
    selected = sorted(
        (a for a in published_advisories(advisories) if fixed_in(a, tag)),
        key=lambda a: a["ghsa_id"],
    )
    if not selected:
        raise ValueError(
            "No published advisories match this fixed release; notes unchanged"
        )
    lines = [START, "## Security fixes", ""]
    for advisory in selected:
        ghsa = advisory["ghsa_id"]
        line = f"- [{ghsa}]({BASE_URL}/security/advisories/{ghsa}): "
        line += inline_text(advisory["summary"])
        cve = advisory.get("cve_id")
        if cve and re.fullmatch(r"CVE-\d{4}-\d{4,}", cve):
            line += f" ({cve})"
        researchers = accepted_researchers(advisory)
        if researchers:
            line += ". Thanks to " + ", ".join(
                f"[@{login}](https://github.com/{login})" for login in researchers
            )
        lines.append(line + ".")
    lines += ["", END]
    section = "\n".join(lines)
    if START in existing or END in existing:
        if existing.count(START) != 1 or existing.count(END) != 1:
            raise ValueError(
                "Expected exactly one complete security-advisories marker pair"
            )
        start, end = existing.index(START), existing.index(END)
        if end < start:
            raise ValueError("Security-advisories markers are reversed")
        return existing[:start] + section + existing[end + len(END) :]
    return existing.rstrip() + ("\n\n" if existing.strip() else "") + section + "\n"


def fetch_advisories():
    result = subprocess.run(
        [
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{REPOSITORY}/security-advisories?state=published&per_page=100",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [advisory for page in json.loads(result.stdout) for advisory in page]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--notes", type=Path, help="Existing release notes to preserve")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existing = args.notes.read_text(encoding="utf-8") if args.notes else ""
    try:
        result = release_notes(fetch_advisories(), args.tag, existing)
    except ValueError as error:
        parser.error(str(error))
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
