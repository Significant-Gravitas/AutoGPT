"""Bounded, anonymous reads from this repository's public GitHub API."""

import json
import re
from urllib.parse import parse_qs, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

REPOSITORY = "Significant-Gravitas/AutoGPT"
REPOSITORY_ID = 614765452
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}/"
MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 20


class PublicDataError(ValueError):
    pass


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PublicDataError("Public API redirects are not allowed")


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
        raise PublicDataError("Invalid public API path")
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
                raise PublicDataError("Invalid public API response")
            return json.loads(raw), response.headers
    except Exception:
        raise PublicDataError(
            "Public API read failed; no authenticated fallback"
        ) from None


def public_pages(endpoint, query=""):
    path = f"{endpoint}?per_page=100" + (f"&{query}" if query else "")
    seen = set()
    result = []
    for _ in range(MAX_PAGES):
        if path in seen:
            raise PublicDataError("Public API pagination repeated a page")
        seen.add(path)
        page, headers = public_json(path)
        if not isinstance(page, list):
            raise PublicDataError("Invalid public API collection")
        result.extend(page)
        matches = re.findall(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
        if not matches:
            return result
        if len(matches) != 1:
            raise PublicDataError("Invalid public API pagination")
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
            raise PublicDataError("Invalid public API pagination target")
        path = endpoint + "?" + target.query
    raise PublicDataError("Public API pagination limit exceeded")
