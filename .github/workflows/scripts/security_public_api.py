"""Bounded, anonymous reads from this repository's public GitHub API."""

import json
import re
from enum import Enum
from urllib.parse import parse_qs, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

REPOSITORY = "Significant-Gravitas/AutoGPT"
REPOSITORY_ID = 614765452
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}/"
MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 20


class PublicErrorCode(Enum):
    API_REDIRECT = "Public API redirects are not allowed"
    API_PATH = "Invalid public API path"
    API_RESPONSE = "Invalid public API response"
    API_READ = "Public API read failed; no authenticated fallback"
    API_COLLECTION = "Invalid public API collection"
    PAGE_REPEATED = "Public API pagination repeated a page"
    PAGE_LINK = "Invalid public API pagination"
    PAGE_TARGET = "Invalid public API pagination target"
    PAGE_LIMIT = "Public API pagination limit exceeded"
    SNAPSHOT_SCHEMA = "Invalid public snapshot schema"
    SNAPSHOT_RECORD = "Invalid public snapshot record"
    SNAPSHOT_FIELD = "Invalid public snapshot field"
    ADVISORY_RECORD = "Invalid public advisory record"
    ADVISORY_ID = "Invalid published advisory identifier"
    ADVISORIES_CHANGED = "Public advisories changed; rerun the workflow"
    MARKER_PAIR = "Expected exactly one complete managed marker pair"
    MARKER_ORDER = "Managed markers are reversed"
    RELEASE_TAG = "Unsupported release tag"
    RELEASE_MAPPING = "Published advisory release mapping is invalid or ambiguous"
    RELEASE_TARGET = "Invalid release write target"
    RELEASE_UPDATE = "Release update failed; rerun after checking public notes"
    RELEASE_CHANGED = "Release changed; rerun the workflow"
    LATEST_RELEASE = "Could not identify the latest release; no notes were updated"
    LATEST_CHANGED = "The latest release changed during the update; check it"
    WRITE_CONTEXT = "Writes require the trusted master workflow"
    FILE_SYMLINK = "Managed files must not be symlinks"
    UNKNOWN = "Unexpected public-data failure; no raw response or report was logged"


class PublicDataError(ValueError):
    def __init__(self, code):
        self.code = (
            code if isinstance(code, PublicErrorCode) else PublicErrorCode.UNKNOWN
        )
        super().__init__(self.code.value)


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PublicDataError(PublicErrorCode.API_REDIRECT)


def public_json(path):
    url = API_ROOT + path
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.github.com"
        or not parsed.path.startswith(f"/repos/{REPOSITORY}/")
        or parsed.fragment
        or ".." in parsed.path
    ):
        raise PublicDataError(PublicErrorCode.API_PATH)
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "AutoGPT-public-security-sync",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    # Do not inherit proxies, netrc, gh credentials, auth handlers, or redirects.
    opener = build_opener(ProxyHandler({}), NoRedirects())
    try:
        with opener.open(request, timeout=20) as response:
            raw = response.read(MAX_BYTES + 1)
            if response.status != 200 or len(raw) > MAX_BYTES:
                raise PublicDataError(PublicErrorCode.API_RESPONSE)
            return json.loads(raw), response.headers
    except PublicDataError:
        raise
    except Exception:
        raise PublicDataError(PublicErrorCode.API_READ) from None


def public_pages(endpoint, query=""):
    path = f"{endpoint}?per_page=100" + (f"&{query}" if query else "")
    seen = set()
    result = []
    for _ in range(MAX_PAGES):
        if path in seen:
            raise PublicDataError(PublicErrorCode.PAGE_REPEATED)
        seen.add(path)
        page, headers = public_json(path)
        if not isinstance(page, list) or any(
            not isinstance(item, dict) for item in page
        ):
            raise PublicDataError(PublicErrorCode.API_COLLECTION)
        result.extend(page)
        matches = re.findall(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        if not matches:
            return result
        if len(matches) != 1:
            raise PublicDataError(PublicErrorCode.PAGE_LINK)
        target = urlsplit(matches[0])
        parameters = parse_qs(target.query)
        if (
            target.scheme != "https"
            or target.netloc != "api.github.com"
            or target.path
            not in {
                urlsplit(API_ROOT + endpoint).path,
                f"/repositories/{REPOSITORY_ID}/{endpoint}",
            }
            or target.fragment
            or parameters.get("per_page") != ["100"]
            or not set(parameters) <= {"per_page", "page", "after", "before", "state"}
            or (query and parameters.get("state") != ["published"])
        ):
            raise PublicDataError(PublicErrorCode.PAGE_TARGET)
        path = endpoint + "?" + target.query
    raise PublicDataError(PublicErrorCode.PAGE_LIMIT)
