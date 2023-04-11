import asyncio
import contextlib
import datetime
import os  # noqa
import pathlib
import pickle
import re
from collections import defaultdict
from http.cookies import BaseCookie, Morsel, SimpleCookie
from typing import (  # noqa
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from yarl import URL

from .abc import AbstractCookieJar, ClearCookiePredicate
from .helpers import is_ip_address, next_whole_second
from .typedefs import LooseCookies, PathLike, StrOrURL

__all__ = ("CookieJar", "DummyCookieJar")


CookieItem = Union[str, "Morsel[str]"]


class CookieJar(AbstractCookieJar):
    """Implements cookie storage adhering to RFC 6265."""

    DATE_TOKENS_RE = re.compile(
        r"[\x09\x20-\x2F\x3B-\x40\x5B-\x60\x7B-\x7E]*"
        r"(?P<token>[\x00-\x08\x0A-\x1F\d:a-zA-Z\x7F-\xFF]+)"
    )

    DATE_HMS_TIME_RE = re.compile(r"(\d{1,2}):(\d{1,2}):(\d{1,2})")

    DATE_DAY_OF_MONTH_RE = re.compile(r"(\d{1,2})")

    DATE_MONTH_RE = re.compile(
        "(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|" "(aug)|(sep)|(oct)|(nov)|(dec)",
        re.I,
    )

    DATE_YEAR_RE = re.compile(r"(\d{2,4})")

    MAX_TIME = datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)

    MAX_32BIT_TIME = datetime.datetime.utcfromtimestamp(2**31 - 1)

    def __init__(
        self,
        *,
        unsafe: bool = False,
        quote_cookie: bool = True,
        treat_as_secure_origin: Union[StrOrURL, List[StrOrURL], None] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        super().__init__(loop=loop)
        self._cookies: DefaultDict[Tuple[str, str], SimpleCookie[str]] = defaultdict(
            SimpleCookie
        )
        self._host_only_cookies: Set[Tuple[str, str]] = set()
        self._unsafe = unsafe
        self._quote_cookie = quote_cookie
        if treat_as_secure_origin is None:
            treat_as_secure_origin = []
        elif isinstance(treat_as_secure_origin, URL):
            treat_as_secure_origin = [treat_as_secure_origin.origin()]
        elif isinstance(treat_as_secure_origin, str):
            treat_as_secure_origin = [URL(treat_as_secure_origin).origin()]
        else:
            treat_as_secure_origin = [
                URL(url).origin() if isinstance(url, str) else url.origin()
                for url in treat_as_secure_origin
            ]
        self._treat_as_secure_origin = treat_as_secure_origin
        self._next_expiration = next_whole_second()
        self._expirations: Dict[Tuple[str, str, str], datetime.datetime] = {}
        # #4515: datetime.max may not be representable on 32-bit platforms
        self._max_time = self.MAX_TIME
        try:
            self._max_time.timestamp()
        except OverflowError:
            self._max_time = self.MAX_32BIT_TIME

    def save(self, file_path: PathLike) -> None:
        file_path = pathlib.Path(file_path)
        with file_path.open(mode="wb") as f:
            pickle.dump(self._cookies, f, pickle.HIGHEST_PROTOCOL)

    def load(self, file_path: PathLike) -> None:
        file_path = pathlib.Path(file_path)
        with file_path.open(mode="rb") as f:
            self._cookies = pickle.load(f)

    def clear(self, predicate: Optional[ClearCookiePredicate] = None) -> None:
        if predicate is None:
            self._next_expiration = next_whole_second()
            self._cookies.clear()
            self._host_only_cookies.clear()
            self._expirations.clear()
            return

        to_del = []
        now = datetime.datetime.now(datetime.timezone.utc)
        for (domain, path), cookie in self._cookies.items():
            for name, morsel in cookie.items():
                key = (domain, path, name)
                if (
                    key in self._expirations and self._expirations[key] <= now
                ) or predicate(morsel):
                    to_del.append(key)

        for domain, path, name in to_del:
            self._host_only_cookies.discard((domain, name))
            key = (domain, path, name)
            if key in self._expirations:
                del self._expirations[(domain, path, name)]
            self._cookies[(domain, path)].pop(name, None)

        next_expiration = min(self._expirations.values(), default=self._max_time)
        try:
            self._next_expiration = next_expiration.replace(
                microsecond=0
            ) + datetime.timedelta(seconds=1)
        except OverflowError:
            self._next_expiration = self._max_time

    def clear_domain(self, domain: str) -> None:
        self.clear(lambda x: self._is_domain_match(domain, x["domain"]))

    def __iter__(self) -> "Iterator[Morsel[str]]":
        self._do_expiration()
        for val in self._cookies.values():
            yield from val.values()

    def __len__(self) -> int:
        return sum(1 for i in self)

    def _do_expiration(self) -> None:
        self.clear(lambda x: False)

    def _expire_cookie(
        self, when: datetime.datetime, domain: str, path: str, name: str
    ) -> None:
        self._next_expiration = min(self._next_expiration, when)
        self._expirations[(domain, path, name)] = when

    def update_cookies(self, cookies: LooseCookies, response_url: URL = URL()) -> None:
        """Update cookies."""
        hostname = response_url.raw_host

        if not self._unsafe and is_ip_address(hostname):
            # Don't accept cookies from IPs
            return

        if isinstance(cookies, Mapping):
            cookies = cookies.items()

        for name, cookie in cookies:
            if not isinstance(cookie, Morsel):
                tmp: SimpleCookie[str] = SimpleCookie()
                tmp[name] = cookie  # type: ignore[assignment]
                cookie = tmp[name]

            domain = cookie["domain"]

            # ignore domains with trailing dots
            if domain.endswith("."):
                domain = ""
                del cookie["domain"]

            if not domain and hostname is not None:
                # Set the cookie's domain to the response hostname
                # and set its host-only-flag
                self._host_only_cookies.add((hostname, name))
                domain = cookie["domain"] = hostname

            if domain.startswith("."):
                # Remove leading dot
                domain = domain[1:]
                cookie["domain"] = domain

            if hostname and not self._is_domain_match(domain, hostname):
                # Setting cookies for different domains is not allowed
                continue

            path = cookie["path"]
            if not path or not path.startswith("/"):
                # Set the cookie's path to the response path
                path = response_url.path
                if not path.startswith("/"):
                    path = "/"
                else:
                    # Cut everything from the last slash to the end
                    path = "/" + path[1 : path.rfind("/")]
                cookie["path"] = path

            max_age = cookie["max-age"]
            if max_age:
                try:
                    delta_seconds = int(max_age)
                    try:
                        max_age_expiration = datetime.datetime.now(
                            datetime.timezone.utc
                        ) + datetime.timedelta(seconds=delta_seconds)
                    except OverflowError:
                        max_age_expiration = self._max_time
                    self._expire_cookie(max_age_expiration, domain, path, name)
                except ValueError:
                    cookie["max-age"] = ""

            else:
                expires = cookie["expires"]
                if expires:
                    expire_time = self._parse_date(expires)
                    if expire_time:
                        self._expire_cookie(expire_time, domain, path, name)
                    else:
                        cookie["expires"] = ""

            self._cookies[(domain, path)][name] = cookie

        self._do_expiration()

    def filter_cookies(
        self, request_url: URL = URL()
    ) -> Union["BaseCookie[str]", "SimpleCookie[str]"]:
        """Returns this jar's cookies filtered by their attributes."""
        self._do_expiration()
        request_url = URL(request_url)
        filtered: Union["SimpleCookie[str]", "BaseCookie[str]"] = (
            SimpleCookie() if self._quote_cookie else BaseCookie()
        )
        hostname = request_url.raw_host or ""
        request_origin = URL()
        with contextlib.suppress(ValueError):
            request_origin = request_url.origin()

        is_not_secure = (
            request_url.scheme not in ("https", "wss")
            and request_origin not in self._treat_as_secure_origin
        )

        for cookie in self:
            name = cookie.key
            domain = cookie["domain"]

            # Send shared cookies
            if not domain:
                filtered[name] = cookie.value
                continue

            if not self._unsafe and is_ip_address(hostname):
                continue

            if (domain, name) in self._host_only_cookies:
                if domain != hostname:
                    continue
            elif not self._is_domain_match(domain, hostname):
                continue

            if not self._is_path_match(request_url.path, cookie["path"]):
                continue

            if is_not_secure and cookie["secure"]:
                continue

            # It's critical we use the Morsel so the coded_value
            # (based on cookie version) is preserved
            mrsl_val = cast("Morsel[str]", cookie.get(cookie.key, Morsel()))
            mrsl_val.set(cookie.key, cookie.value, cookie.coded_value)
            filtered[name] = mrsl_val

        return filtered

    @staticmethod
    def _is_domain_match(domain: str, hostname: str) -> bool:
        """Implements domain matching adhering to RFC 6265."""
        if hostname == domain:
            return True

        if not hostname.endswith(domain):
            return False

        non_matching = hostname[: -len(domain)]

        if not non_matching.endswith("."):
            return False

        return not is_ip_address(hostname)

    @staticmethod
    def _is_path_match(req_path: str, cookie_path: str) -> bool:
        """Implements path matching adhering to RFC 6265."""
        if not req_path.startswith("/"):
            req_path = "/"

        if req_path == cookie_path:
            return True

        if not req_path.startswith(cookie_path):
            return False

        if cookie_path.endswith("/"):
            return True

        non_matching = req_path[len(cookie_path) :]

        return non_matching.startswith("/")

    @classmethod
    def _parse_date(cls, date_str: str) -> Optional[datetime.datetime]:
        """Implements date string parsing adhering to RFC 6265."""
        if not date_str:
            return None

        found_time = False
        found_day = False
        found_month = False
        found_year = False

        hour = minute = second = 0
        day = 0
        month = 0
        year = 0

        for token_match in cls.DATE_TOKENS_RE.finditer(date_str):

            token = token_match.group("token")

            if not found_time:
                time_match = cls.DATE_HMS_TIME_RE.match(token)
                if time_match:
                    found_time = True
                    hour, minute, second = (int(s) for s in time_match.groups())
                    continue

            if not found_day:
                day_match = cls.DATE_DAY_OF_MONTH_RE.match(token)
                if day_match:
                    found_day = True
                    day = int(day_match.group())
                    continue

            if not found_month:
                month_match = cls.DATE_MONTH_RE.match(token)
                if month_match:
                    found_month = True
                    assert month_match.lastindex is not None
                    month = month_match.lastindex
                    continue

            if not found_year:
                year_match = cls.DATE_YEAR_RE.match(token)
                if year_match:
                    found_year = True
                    year = int(year_match.group())

        if 70 <= year <= 99:
            year += 1900
        elif 0 <= year <= 69:
            year += 2000

        if False in (found_day, found_month, found_year, found_time):
            return None

        if not 1 <= day <= 31:
            return None

        if year < 1601 or hour > 23 or minute > 59 or second > 59:
            return None

        return datetime.datetime(
            year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc
        )


class DummyCookieJar(AbstractCookieJar):
    """Implements a dummy cookie storage.

    It can be used with the ClientSession when no cookie processing is needed.

    """

    def __init__(self, *, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        super().__init__(loop=loop)

    def __iter__(self) -> "Iterator[Morsel[str]]":
        while False:
            yield None

    def __len__(self) -> int:
        return 0

    def clear(self, predicate: Optional[ClearCookiePredicate] = None) -> None:
        pass

    def clear_domain(self, domain: str) -> None:
        pass

    def update_cookies(self, cookies: LooseCookies, response_url: URL = URL()) -> None:
        pass

    def filter_cookies(self, request_url: URL) -> "BaseCookie[str]":
        return SimpleCookie()
