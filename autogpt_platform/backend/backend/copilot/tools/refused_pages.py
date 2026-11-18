"""Recognise a page that refused an automated visitor, so the tool says so and
names a fallback instead of handing the model an empty page or a sign-in form."""

import re
from urllib.parse import urlparse

# Run in the browser page. Cloudflare's challenge sets ``_cf_chl_opt``; its
# title is localised, so the title cannot be the marker.
PAGE_CHECK_JS = (
    "window._cf_chl_opt ? 'bot_check'"
    " : document.querySelector('input[type=password]') ? 'sign_in_form' : ''"
)

# Sites whose pages an anonymous fetch gets empty or behind a sign-in form.
_SIGN_IN_ONLY_SITES = {
    "facebook.com": "Facebook",
    "instagram.com": "Instagram",
    "linkedin.com": "LinkedIn",
    "x.com": "X",
    "twitter.com": "X",
}
_SIGN_IN_PATH = re.compile(
    r"/(?:login(?:\.php)?|checkpoint|authwall|accounts/login|i/flow/login)(?:/|$)"
)


def refused_page_notice(
    url: str, *, bot_check: bool = False, content_withheld: bool = False
) -> str | None:
    """The notice for a page that refused an automated visitor, or None.

    ``content_withheld``: the page came back with no readable text, as a
    sign-in form or as an error status; it counts only on a site in
    ``_SIGN_IN_ONLY_SITES``.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").removeprefix("www.")
    if bot_check:
        return (
            f"{host} answered with a bot check, so this page can't be read "
            "automatically; don't retry it. Use web_search for it or another "
            "source, and tell the user in one sentence what you used instead."
        )
    site = _sign_in_only_site(host)
    if site and (content_withheld or _SIGN_IN_PATH.match(parsed.path)):
        return (
            f"{site} shows this page only to signed-in visitors; unless you are "
            "signing in for the user, don't retry it. Use web_search for this "
            "page or the business's own website, and tell the user in one "
            "sentence what you used instead."
        )
    return None


def _sign_in_only_site(host: str) -> str | None:
    for domain, name in _SIGN_IN_ONLY_SITES.items():
        if host == domain or host.endswith(f".{domain}"):
            return name
    return None
