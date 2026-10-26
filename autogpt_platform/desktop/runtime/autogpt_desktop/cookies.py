"""Cookies of several installs in one browser.

Every install is served from 127.0.0.1 on a port of its own, and a browser
keeps cookies per host, not per port. The platform names its cookies the same
in every install. Left alone, signing in to one install in the system browser
would replace the session cookie of the other, and each install would be sent
the other's session cookie with every request: a variant is experimental code,
and must not be handed a credential of the normal app.

So a variant's cookies carry its install name in the browser:
`AutoGPT-voice.<name>`. The proxy adds the mark to every cookie the platform
sets and takes it off again on the way in, and passes on no cookie that is
not marked as this install's. The normal app keeps the plain names, so that
nobody is signed out by an update, and leaves out the cookies marked for a
variant. The platform itself sees its own names throughout.

What this cannot do: a cookie that a page sets with `document.cookie` has no
mark, so a variant's server is not sent it, and a script that reads a cookie
the server set finds it under the marked name. The platform's sign-in uses
neither.
"""

from __future__ import annotations

import re

from autogpt_desktop import install

# Browsers attach rules to names that start with these; a mark goes after.
BROWSER_PREFIXES = ("__Secure-", "__Host-")
VARIANT_MARK = re.compile(r"AutoGPT-[a-z0-9-]+\.")


def mark_of(install_name: str) -> str:
    """What this install's cookie names start with in the browser."""
    return "" if install_name == install.NORMAL else f"{install_name}."


def to_browser(set_cookie: str, mark: str) -> str:
    """A Set-Cookie value from the platform, as the browser is to store it."""
    if not mark:
        return set_cookie
    name, separator, rest = set_cookie.partition("=")
    prefix, plain = _split(name.strip())
    return f"{prefix}{mark}{plain}{separator}{rest}"


def to_platform(cookie: str, mark: str) -> str | None:
    """A Cookie header from the browser, with this install's cookies only and
    under the platform's names. None when none of them is this install's."""
    pairs = [pair.strip() for pair in cookie.split(";") if pair.strip()]
    if not mark and not any(_marked(pair) for pair in pairs):
        return cookie
    own = [named for pair in pairs if (named := _own(pair, mark)) is not None]
    return "; ".join(own) or None


def _own(pair: str, mark: str) -> str | None:
    name, separator, value = pair.partition("=")
    prefix, plain = _split(name.strip())
    if not mark:
        return None if VARIANT_MARK.match(plain) else pair
    if not plain.startswith(mark):
        return None
    return f"{prefix}{plain[len(mark):]}{separator}{value}"


def _marked(pair: str) -> bool:
    return VARIANT_MARK.match(_split(pair.partition("=")[0].strip())[1]) is not None


def _split(name: str) -> tuple[str, str]:
    for prefix in BROWSER_PREFIXES:
        if name.startswith(prefix):
            return prefix, name[len(prefix) :]
    return "", name
