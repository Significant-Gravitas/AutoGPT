"""What GTM segments checkout openers on, worked out from what we hold.

Pure functions, shared by the live checkout event and the backfill so both
write the same values:

- `checkout_opened_date`: the first time they opened Stripe checkout.
- `email_type`: business, personal (a webmail provider) or education.
- `signin_method`: google or email.
- `country` (MailerLite's built-in field, the full English name) and
  `country_code` (ISO 3166-1 alpha-2), from the strongest source we have: the
  Stripe billing address, then the visitor's IP country as the edge saw it,
  then the browser's timezone. `country_source` says which.
- `exclude_de_at`: "yes" when any signal points at Germany or Austria, whose
  stricter marketing rules GTM skips. Once yes it is never set back to no.

The timezone and country tables come from tzdata, which zoneinfo already
depends on. A zone is looked up by the name the browser reported before any
alias is followed: tzdata merges zones with identical history across borders
(Europe/Stockholm is a link to Europe/Berlin), and following that link first
would tag a Swede as German.
"""

import functools
import importlib.resources
from collections.abc import Iterable, Mapping
from datetime import datetime

from backend.data.notifications import SubscriberField, SubscriptionStatus
from backend.notifications.subscriber_fields import Fields, mailerlite_date

YES = "yes"
NO = "no"

DE_AT = frozenset({"DE", "AT"})

# Strongest first. A held country from a stronger source is never replaced by
# a weaker one; a hand-typed one with no source ranks lowest.
COUNTRY_SOURCES = ("stripe", "ip", "timezone")
_SOURCE_RANK = {
    source: len(COUNTRY_SOURCES) - i for i, source in enumerate(COUNTRY_SOURCES)
}

# Zones that name no country.
_COUNTRYLESS = frozenset(
    {
        "", "not-set", "UTC", "UCT", "GMT", "GMT0", "GMT+0", "GMT-0", "Greenwich",
        "Universal", "Zulu", "Factory", "CET", "MET", "EET", "WET", "EST", "MST",
        "HST", "EST5EDT", "CST6CDT", "MST7MDT", "PST8PDT",
    }
)  # fmt: skip

# Old names whose tzdata link points across a border.
_CROSS_BORDER_ALIASES = {
    "Africa/Asmera": "ER",
    "Africa/Timbuktu": "ML",
    "America/Coral_Harbour": "CA",
    "America/Virgin": "VI",
    "Antarctica/South_Pole": "AQ",
    "Atlantic/Jan_Mayen": "SJ",
    "Iceland": "IS",
    "Pacific/Johnston": "UM",
    "Pacific/Ponape": "FM",
    "Pacific/Truk": "FM",
    "Pacific/Yap": "FM",
}

# tzdata's names, where they read oddly in a segment filter.
_NAME_OVERRIDES = {
    "GB": "United Kingdom",
    "KR": "South Korea",
    "KP": "North Korea",
    "MM": "Myanmar",
    "SZ": "Eswatini",
}

# Webmail providers: an address on one is a person's, not a company's.
_PERSONAL_DOMAINS = frozenset(
    {
        "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
        "msn.com", "yahoo.com", "ymail.com", "rocketmail.com", "icloud.com",
        "me.com", "mac.com", "aol.com", "proton.me", "protonmail.com", "pm.me",
        "gmx.com", "gmx.net", "gmx.de", "web.de", "t-online.de", "freenet.de",
        "yandex.com", "yandex.ru", "ya.ru", "mail.ru", "inbox.ru", "list.ru",
        "bk.ru", "rambler.ru", "qq.com", "foxmail.com", "163.com", "126.com",
        "yeah.net", "sina.com", "sohu.com", "aliyun.com", "naver.com",
        "daum.net", "hanmail.net", "kakao.com", "rediffmail.com", "zoho.com",
        "zohomail.com", "tutanota.com", "tuta.io", "fastmail.com", "hey.com",
        "mail.com", "email.com", "seznam.cz", "wp.pl", "o2.pl", "interia.pl",
        "libero.it", "virgilio.it", "orange.fr", "free.fr", "laposte.net",
        "sfr.fr", "btinternet.com", "comcast.net", "verizon.net", "att.net",
        "sbcglobal.net", "bellsouth.net", "cox.net", "shaw.ca", "rogers.com",
        "bigpond.com", "optusnet.com.au", "uol.com.br", "bol.com.br",
        "terra.com.br",
    }
)  # fmt: skip

# Providers that run the same webmail under many country domains.
_PERSONAL_PREFIXES = ("yahoo.", "hotmail.", "outlook.", "live.", "gmx.", "yandex.")


def _tz_file(name: str) -> str:
    return (
        importlib.resources.files("tzdata")
        .joinpath("zoneinfo")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


@functools.cache
def _zone_countries() -> dict[str, str]:
    """Each zone in zone.tab, by the name a browser reports, to its country.
    zone.tab keeps one row per named zone even where tzdata has since merged
    zones, which is why it is used instead of zone1970.tab."""
    zones: dict[str, str] = {}
    for line in _tz_file("zone.tab").splitlines():
        if line and not line.startswith("#"):
            code, _coordinates, zone, *_ = line.split("\t")
            zones[zone] = code
    return zones


@functools.cache
def _zone_links() -> dict[str, str]:
    """Old and alternative zone names to the zone they now point at."""
    links: dict[str, str] = {}
    for line in _tz_file("tzdata.zi").splitlines():
        if line.startswith("L "):
            _, target, alias = line.split()
            links[alias] = target
    return links


@functools.cache
def _country_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for line in _tz_file("iso3166.tab").splitlines():
        if line and not line.startswith("#"):
            code, name = line.split("\t")[:2]
            names[code] = name.strip()
    names.update(_NAME_OVERRIDES)
    return names


def timezone_country(timezone: str | None) -> str | None:
    """The country an IANA timezone name, as the browser reported it, belongs
    to, or None for a zone that names no country."""
    name = (timezone or "").strip()
    if name in _COUNTRYLESS or name.startswith("Etc/"):
        return None
    if name in _CROSS_BORDER_ALIASES:
        return _CROSS_BORDER_ALIASES[name]
    zones = _zone_countries()
    if name in zones:
        return zones[name]
    return zones.get(_zone_links().get(name, ""))


def country_code(value: str | None) -> str | None:
    """An ISO 3166-1 alpha-2 code we know a name for, or None."""
    code = (value or "").strip().upper()
    return code if code in _country_names() else None


def country_name(code: str) -> str:
    return _country_names()[code]


def email_domain(email: str) -> str:
    return email.strip().lower().rpartition("@")[2]


def email_type(email: str) -> str:
    """education for a school or university address, personal for a webmail
    provider's, and business for everything else."""
    domain = email_domain(email)
    labels = domain.split(".")
    if labels[-1] == "edu" or (
        len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in ("edu", "ac")
    ):
        return "education"
    if domain in _PERSONAL_DOMAINS or domain.startswith(_PERSONAL_PREFIXES):
        return "personal"
    return "business"


def email_points_at_de_at(email: str) -> bool:
    """A .de or .at address, whatever sits under the country code (.co.at,
    .ac.at)."""
    return email_domain(email).rsplit(".", 1)[-1] in ("de", "at")


def signin_method(providers: Iterable[str]) -> str | None:
    """How the account signs in, from its auth provider ids: google wins over
    a password, since it proves the address."""
    found = {p.strip().lower() for p in providers if p}
    if "google" in found:
        return "google"
    if found & {"credential", "email"}:
        return "email"
    return min(found) if found else None


def checkout_fields(
    *,
    email: str,
    created_at: datetime,
    opened_at: datetime | int | float | None,
    signin_providers: Iterable[str],
    timezone: str | None,
    stripe_country: str | None = None,
    ip_country: str | None = None,
    status: str = SubscriptionStatus.SIGNED.value,
) -> Fields:
    """Every field a checkout opener gets. Merge it with what MailerLite holds
    (`merge_with_held`) before writing."""
    tz_country = timezone_country(timezone)
    candidates = {
        "stripe": country_code(stripe_country),
        "ip": country_code(ip_country),
        "timezone": tz_country,
    }
    signals = {code for code in candidates.values() if code}
    fields: Fields = {
        SubscriberField.STATUS: status,
        SubscriberField.SIGNUP: mailerlite_date(created_at),
        SubscriberField.CHECKOUT_OPENED: mailerlite_date(opened_at),
        SubscriberField.EMAIL_TYPE: email_type(email),
        SubscriberField.EXCLUDE_DE_AT: (
            YES if signals & DE_AT or email_points_at_de_at(email) else NO
        ),
    }
    method = signin_method(signin_providers)
    if method:
        fields[SubscriberField.SIGNIN_METHOD] = method
    for source in COUNTRY_SOURCES:
        code = candidates[source]
        if code:
            fields[SubscriberField.COUNTRY] = country_name(code)
            fields[SubscriberField.COUNTRY_CODE] = code
            fields[SubscriberField.COUNTRY_SOURCE] = source
            break
    return fields


_COUNTRY_FIELDS = (
    SubscriberField.COUNTRY,
    SubscriberField.COUNTRY_CODE,
    SubscriberField.COUNTRY_SOURCE,
)

# A hand-typed built-in country that names Germany or Austria.
_DE_AT_NAMES = frozenset({"germany", "deutschland", "austria", "österreich"})


def _held_points_at_de_at(held: Mapping[str, object]) -> bool:
    """A country MailerLite already holds is a signal too, whether ours or
    typed by hand, and the exclusion holds for any signal."""
    code = str(held.get(SubscriberField.COUNTRY_CODE.value) or "").strip().upper()
    name = str(held.get(SubscriberField.COUNTRY.value) or "").strip().lower()
    return code in DE_AT or name in _DE_AT_NAMES


def merge_with_held(
    fields: Mapping[SubscriberField, str | None],
    held: Mapping[str, object],
    *,
    keep_held_status: bool = True,
) -> Fields:
    """Drop whatever would make MailerLite's copy worse:

    - a status it already holds (unless the caller knows the true one), since
      `signed` is only where everyone starts;
    - a later checkout_opened_date than the one it holds;
    - exclude_de_at "no" over "yes", or over a German or Austrian country it
      holds;
    - a country from a weaker source than the one it holds.
    """
    merged = dict(fields)
    if keep_held_status and held.get(SubscriberField.STATUS.value):
        merged.pop(SubscriberField.STATUS, None)
    held_opened = str(held.get(SubscriberField.CHECKOUT_OPENED.value) or "")[:10]
    ours_opened = merged.get(SubscriberField.CHECKOUT_OPENED)
    if held_opened and ours_opened and held_opened <= ours_opened:
        merged.pop(SubscriberField.CHECKOUT_OPENED)
    if held.get(SubscriberField.EXCLUDE_DE_AT.value) == YES:
        merged.pop(SubscriberField.EXCLUDE_DE_AT, None)
    elif _held_points_at_de_at(held):
        merged[SubscriberField.EXCLUDE_DE_AT] = YES
    held_rank = _SOURCE_RANK.get(str(held.get(SubscriberField.COUNTRY_SOURCE.value)), 0)
    ours_rank = _SOURCE_RANK.get(str(merged.get(SubscriberField.COUNTRY_SOURCE)), 0)
    if held_rank > ours_rank:
        for field in _COUNTRY_FIELDS:
            merged.pop(field, None)
    return merged
