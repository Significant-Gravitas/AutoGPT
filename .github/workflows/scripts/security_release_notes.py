"""Match published security advisories to the releases that fixed them."""

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


def fixed_in(advisory, tag):
    """Version rows patched in this release, if the description links its exact tag,
    which names the product line (Classic and Platform share version numbers)."""
    version = version_number(tag)
    release_url = f"{BASE_URL}/releases/tag/{tag}"
    # Accept closing delimiters and prose punctuation without matching tag extensions.
    has_reference = re.search(
        re.escape(release_url) + r"""(?=$|[\s)#?<>\]"'`]|[.,;:!]+(?=$|[\s)<>\]"'`]))""",
        advisory.get("description") or "",
    )
    if not version or not has_reference:
        return []
    return [
        row
        for row in advisory.get("vulnerabilities") or []
        if version_number(row.get("patched_versions")) == version
    ]


def fetch_advisories():
    return public_pages("security-advisories", "state=published")
