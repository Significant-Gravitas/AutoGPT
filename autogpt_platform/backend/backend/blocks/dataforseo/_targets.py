"""
Turning what people paste (a bare domain, a home page URL, a page URL) into
the target forms DataForSEO expects.
"""

from urllib.parse import urlparse


def normalise_domain(target: str) -> str:
    """'https://www.example.com/shop' -> 'example.com'."""
    target = target.strip()
    host = urlparse(target if "://" in target else f"//{target}").hostname or ""
    return host.removeprefix("www.")


def normalise_target(target: str) -> str:
    """A domain stays a domain and a page stays a page.

    'https://www.example.com/' -> 'example.com'
    'example.com/shop' -> 'https://example.com/shop'
    """
    target = target.strip()
    parsed = urlparse(target if "://" in target else f"https://{target}")
    if parsed.path in ("", "/") and not parsed.query:
        return normalise_domain(target)
    return parsed.geturl()
