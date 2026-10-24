"""Generate release-note links from published repository security advisories."""

import argparse
import json
import re
from pathlib import Path

from security_public_api import public_pages

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
        and re.fullmatch(
            r"GHSA-[a-z0-9]{4}(?:-[a-z0-9]{4}){2}", advisory.get("ghsa_id") or ""
        )
    ]


def accepted_researchers(advisory):
    return sorted(
        {
            login
            for credit in advisory.get("credits_detailed") or []
            if credit.get("state") == "accepted"
            and credit.get("type") in RESEARCH_ROLES
            and (login := (credit.get("user") or {}).get("login"))
            and re.fullmatch(r"[A-Za-z0-9-]+", login)
        },
        key=str.casefold,
    )


def version_number(value):
    match = re.fullmatch(
        r"(?:>=\s*)?(?:autogpt-platform-beta-v|agpt-platform-beta-v|autogpt-v|v)?"
        r"(\d+\.\d+\.\d+)",
        (value or "").strip(),
    )
    return match[1] if match else None


def release_packages(advisory, tag):
    packages = []
    for raw in re.findall(
        r"<!-- security-release:\s*(.*?)\s*-->",
        advisory.get("description") or "",
        re.DOTALL,
    ):
        try:
            reference = json.loads(raw)
            package = reference["package"]
            if not isinstance(reference["tag"], str) or not all(
                isinstance(package[key], str) and package[key]
                for key in ("ecosystem", "name")
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"Invalid package-specific release reference in {advisory['ghsa_id']}"
            ) from error
        if reference["tag"] == tag:
            packages.append(package)
    return packages


def fixed_in(advisory, tag):
    version = version_number(tag)
    if not version:
        raise ValueError(
            "Use a stable release tag, such as autogpt-platform-beta-v0.9.0"
        )
    release_url = f"{BASE_URL}/releases/tag/{tag}"
    # Accept closing delimiters and prose punctuation without matching tag extensions.
    has_reference = re.search(
        re.escape(release_url) + r"""(?=$|[\s)#?<>\]"'`]|[.,;:!]+(?=$|[\s)<>\]"'`]))""",
        advisory.get("description") or "",
    )
    vulnerabilities = advisory.get("vulnerabilities") or []
    matching = [
        vulnerability
        for vulnerability in vulnerabilities
        if version_number(vulnerability.get("patched_versions")) == version
    ]
    if not has_reference or not matching:
        return False
    if len(vulnerabilities) > 1:
        packages = release_packages(advisory, tag)
        if not packages:
            raise ValueError(
                f"{advisory['ghsa_id']} needs a package-specific release reference "
                f"for {tag}; notes unchanged"
            )
        matching = [
            v
            for v in matching
            if any(
                all(
                    (v.get("package") or {}).get(key) == p[key]
                    for key in ("ecosystem", "name")
                )
                for p in packages
            )
        ]
    return bool(matching)


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
    return public_pages("security-advisories", "state=published")


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
