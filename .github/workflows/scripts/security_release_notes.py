"""Match published security advisories to the releases that fixed them."""

import json
import re

from security_public_api import public_pages

REPOSITORY = "Significant-Gravitas/AutoGPT"
BASE_URL = f"https://github.com/{REPOSITORY}"
START = "<!-- security-advisories:start -->"
END = "<!-- security-advisories:end -->"
RESEARCH_ROLES = {"reporter", "finder", "analyst"}
LOGIN_PATTERN = r"[A-Za-z0-9][A-Za-z0-9-]{0,38}"
# Our own maintainers can hold research credits on our advisories (an internal find,
# a triage analysis); they aren't outside researchers. Add one when that happens.
TEAM_LOGINS = frozenset(
    login.casefold()
    for login in ("Abhi1992002", "Bentlybro", "kcze", "ntindle", "Pwuts", "Torantulino")
)


def accepted_researchers(advisory):
    return sorted(
        {
            login
            for credit in advisory.get("credits_detailed") or []
            if credit.get("state") == "accepted"
            and credit.get("type") in RESEARCH_ROLES
            and isinstance(login := (credit.get("user") or {}).get("login"), str)
            and re.fullmatch(LOGIN_PATTERN, login)
            and login.casefold() not in TEAM_LOGINS
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
    # Rows that agree on the fixed version (e.g. the Platform source and its Docker image,
    # both patched in 0.9.0) can't be misattributed; only disagreeing rows need a mapping.
    patched = {version_number(v.get("patched_versions")) for v in vulnerabilities}
    if len(patched - {None}) > 1:
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


def fetch_advisories():
    return public_pages("security-advisories", "state=published")
