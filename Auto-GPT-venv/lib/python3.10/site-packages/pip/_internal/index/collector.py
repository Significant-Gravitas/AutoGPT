"""
The main purpose of this module is to expose LinkCollector.collect_sources().
"""

import cgi
import collections
import functools
import itertools
import logging
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree
from html.parser import HTMLParser
from optparse import Values
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pip._vendor import html5lib, requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError

from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import pairwise, redact_auth_from_url
from pip._internal.vcs import vcs

from .sources import CandidatesFromPage, LinkSource, build_source

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

logger = logging.getLogger(__name__)

HTMLElement = xml.etree.ElementTree.Element
ResponseHeaders = MutableMapping[str, str]


def _match_vcs_scheme(url: str) -> Optional[str]:
    """Look for VCS schemes in the URL.

    Returns the matched VCS scheme, or None if there's no match.
    """
    for scheme in vcs.schemes:
        if url.lower().startswith(scheme) and url[len(scheme)] in "+:":
            return scheme
    return None


class _NotHTML(Exception):
    def __init__(self, content_type: str, request_desc: str) -> None:
        super().__init__(content_type, request_desc)
        self.content_type = content_type
        self.request_desc = request_desc


def _ensure_html_header(response: Response) -> None:
    """Check the Content-Type header to ensure the response contains HTML.

    Raises `_NotHTML` if the content type is not text/html.
    """
    content_type = response.headers.get("Content-Type", "")
    if not content_type.lower().startswith("text/html"):
        raise _NotHTML(content_type, response.request.method)


class _NotHTTP(Exception):
    pass


def _ensure_html_response(url: str, session: PipSession) -> None:
    """Send a HEAD request to the URL, and ensure the response contains HTML.

    Raises `_NotHTTP` if the URL is not available for a HEAD request, or
    `_NotHTML` if the content type is not text/html.
    """
    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(url)
    if scheme not in {"http", "https"}:
        raise _NotHTTP()

    resp = session.head(url, allow_redirects=True)
    raise_for_status(resp)

    _ensure_html_header(resp)


def _get_html_response(url: str, session: PipSession) -> Response:
    """Access an HTML page with GET, and return the response.

    This consists of three parts:

    1. If the URL looks suspiciously like an archive, send a HEAD first to
       check the Content-Type is HTML, to avoid downloading a large file.
       Raise `_NotHTTP` if the content type cannot be determined, or
       `_NotHTML` if it is not HTML.
    2. Actually perform the request. Raise HTTP exceptions on network failures.
    3. Check the Content-Type header to make sure we got HTML, and raise
       `_NotHTML` otherwise.
    """
    if is_archive_file(Link(url).filename):
        _ensure_html_response(url, session=session)

    logger.debug("Getting page %s", redact_auth_from_url(url))

    resp = session.get(
        url,
        headers={
            "Accept": "text/html",
            # We don't want to blindly returned cached data for
            # /simple/, because authors generally expecting that
            # twine upload && pip install will function, but if
            # they've done a pip install in the last ~10 minutes
            # it won't. Thus by setting this to zero we will not
            # blindly use any cached data, however the benefit of
            # using max-age=0 instead of no-cache, is that we will
            # still support conditional requests, so we will still
            # minimize traffic sent in cases where the page hasn't
            # changed at all, we will just always incur the round
            # trip for the conditional GET now instead of only
            # once per 10 minutes.
            # For more information, please see pypa/pip#5670.
            "Cache-Control": "max-age=0",
        },
    )
    raise_for_status(resp)

    # The check for archives above only works if the url ends with
    # something that looks like an archive. However that is not a
    # requirement of an url. Unless we issue a HEAD request on every
    # url we cannot know ahead of time for sure if something is HTML
    # or not. However we can check after we've downloaded it.
    _ensure_html_header(resp)

    return resp


def _get_encoding_from_headers(headers: ResponseHeaders) -> Optional[str]:
    """Determine if we have any encoding information in our headers."""
    if headers and "Content-Type" in headers:
        content_type, params = cgi.parse_header(headers["Content-Type"])
        if "charset" in params:
            return params["charset"]
    return None


def _determine_base_url(document: HTMLElement, page_url: str) -> str:
    """Determine the HTML document's base URL.

    This looks for a ``<base>`` tag in the HTML document. If present, its href
    attribute denotes the base URL of anchor tags in the document. If there is
    no such tag (or if it does not have a valid href attribute), the HTML
    file's URL is used as the base URL.

    :param document: An HTML document representation. The current
        implementation expects the result of ``html5lib.parse()``.
    :param page_url: The URL of the HTML document.

    TODO: Remove when `html5lib` is dropped.
    """
    for base in document.findall(".//base"):
        href = base.get("href")
        if href is not None:
            return href
    return page_url


def _clean_url_path_part(part: str) -> str:
    """
    Clean a "part" of a URL path (i.e. after splitting on "@" characters).
    """
    # We unquote prior to quoting to make sure nothing is double quoted.
    return urllib.parse.quote(urllib.parse.unquote(part))


def _clean_file_url_path(part: str) -> str:
    """
    Clean the first part of a URL path that corresponds to a local
    filesystem path (i.e. the first part after splitting on "@" characters).
    """
    # We unquote prior to quoting to make sure nothing is double quoted.
    # Also, on Windows the path part might contain a drive letter which
    # should not be quoted. On Linux where drive letters do not
    # exist, the colon should be quoted. We rely on urllib.request
    # to do the right thing here.
    return urllib.request.pathname2url(urllib.request.url2pathname(part))


# percent-encoded:                   /
_reserved_chars_re = re.compile("(@|%2F)", re.IGNORECASE)


def _clean_url_path(path: str, is_local_path: bool) -> str:
    """
    Clean the path portion of a URL.
    """
    if is_local_path:
        clean_func = _clean_file_url_path
    else:
        clean_func = _clean_url_path_part

    # Split on the reserved characters prior to cleaning so that
    # revision strings in VCS URLs are properly preserved.
    parts = _reserved_chars_re.split(path)

    cleaned_parts = []
    for to_clean, reserved in pairwise(itertools.chain(parts, [""])):
        cleaned_parts.append(clean_func(to_clean))
        # Normalize %xx escapes (e.g. %2f -> %2F)
        cleaned_parts.append(reserved.upper())

    return "".join(cleaned_parts)


def _clean_link(url: str) -> str:
    """
    Make sure a link is fully quoted.
    For example, if ' ' occurs in the URL, it will be replaced with "%20",
    and without double-quoting other characters.
    """
    # Split the URL into parts according to the general structure
    # `scheme://netloc/path;parameters?query#fragment`.
    result = urllib.parse.urlparse(url)
    # If the netloc is empty, then the URL refers to a local filesystem path.
    is_local_path = not result.netloc
    path = _clean_url_path(result.path, is_local_path=is_local_path)
    return urllib.parse.urlunparse(result._replace(path=path))


def _create_link_from_element(
    element_attribs: Dict[str, Optional[str]],
    page_url: str,
    base_url: str,
) -> Optional[Link]:
    """
    Convert an anchor element's attributes in a simple repository page to a Link.
    """
    href = element_attribs.get("href")
    if not href:
        return None

    url = _clean_link(urllib.parse.urljoin(base_url, href))
    pyrequire = element_attribs.get("data-requires-python")
    yanked_reason = element_attribs.get("data-yanked")

    link = Link(
        url,
        comes_from=page_url,
        requires_python=pyrequire,
        yanked_reason=yanked_reason,
    )

    return link


class CacheablePageContent:
    def __init__(self, page: "HTMLPage") -> None:
        assert page.cache_link_parsing
        self.page = page

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.page.url == other.page.url

    def __hash__(self) -> int:
        return hash(self.page.url)


class ParseLinks(Protocol):
    def __call__(
        self, page: "HTMLPage", use_deprecated_html5lib: bool
    ) -> Iterable[Link]:
        ...


def with_cached_html_pages(fn: ParseLinks) -> ParseLinks:
    """
    Given a function that parses an Iterable[Link] from an HTMLPage, cache the
    function's result (keyed by CacheablePageContent), unless the HTMLPage
    `page` has `page.cache_link_parsing == False`.
    """

    @functools.lru_cache(maxsize=None)
    def wrapper(
        cacheable_page: CacheablePageContent, use_deprecated_html5lib: bool
    ) -> List[Link]:
        return list(fn(cacheable_page.page, use_deprecated_html5lib))

    @functools.wraps(fn)
    def wrapper_wrapper(page: "HTMLPage", use_deprecated_html5lib: bool) -> List[Link]:
        if page.cache_link_parsing:
            return wrapper(CacheablePageContent(page), use_deprecated_html5lib)
        return list(fn(page, use_deprecated_html5lib))

    return wrapper_wrapper


def _parse_links_html5lib(page: "HTMLPage") -> Iterable[Link]:
    """
    Parse an HTML document, and yield its anchor elements as Link objects.

    TODO: Remove when `html5lib` is dropped.
    """
    document = html5lib.parse(
        page.content,
        transport_encoding=page.encoding,
        namespaceHTMLElements=False,
    )

    url = page.url
    base_url = _determine_base_url(document, url)
    for anchor in document.findall(".//a"):
        link = _create_link_from_element(
            anchor.attrib,
            page_url=url,
            base_url=base_url,
        )
        if link is None:
            continue
        yield link


@with_cached_html_pages
def parse_links(page: "HTMLPage", use_deprecated_html5lib: bool) -> Iterable[Link]:
    """
    Parse an HTML document, and yield its anchor elements as Link objects.
    """
    encoding = page.encoding or "utf-8"

    # Check if the page starts with a valid doctype, to decide whether to use
    # http.parser or (deprecated) html5lib for parsing -- unless explicitly
    # requested to use html5lib.
    if not use_deprecated_html5lib:
        expected_doctype = "<!doctype html>".encode(encoding)
        actual_start = page.content[: len(expected_doctype)]
        if actual_start.decode(encoding).lower() != "<!doctype html>":
            deprecated(
                reason=(
                    f"The HTML index page being used ({page.url}) is not a proper "
                    "HTML 5 document. This is in violation of PEP 503 which requires "
                    "these pages to be well-formed HTML 5 documents. Please reach out "
                    "to the owners of this index page, and ask them to update this "
                    "index page to a valid HTML 5 document."
                ),
                replacement=None,
                gone_in="22.2",
                issue=10825,
            )
            use_deprecated_html5lib = True

    if use_deprecated_html5lib:
        yield from _parse_links_html5lib(page)
        return

    parser = HTMLLinkParser()
    parser.feed(page.content.decode(encoding))

    url = page.url
    base_url = parser.base_url or url
    for anchor in parser.anchors:
        link = _create_link_from_element(
            anchor,
            page_url=url,
            base_url=base_url,
        )
        if link is None:
            continue
        yield link


class HTMLPage:
    """Represents one page, along with its URL"""

    def __init__(
        self,
        content: bytes,
        encoding: Optional[str],
        url: str,
        cache_link_parsing: bool = True,
    ) -> None:
        """
        :param encoding: the encoding to decode the given content.
        :param url: the URL from which the HTML was downloaded.
        :param cache_link_parsing: whether links parsed from this page's url
                                   should be cached. PyPI index urls should
                                   have this set to False, for example.
        """
        self.content = content
        self.encoding = encoding
        self.url = url
        self.cache_link_parsing = cache_link_parsing

    def __str__(self) -> str:
        return redact_auth_from_url(self.url)


class HTMLLinkParser(HTMLParser):
    """
    HTMLParser that keeps the first base HREF and a list of all anchor
    elements' attributes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._seen_decl = False
        self.base_url: Optional[str] = None
        self.anchors: List[Dict[str, Optional[str]]] = []

    def handle_decl(self, decl: str) -> None:
        if decl.lower() != "doctype html":
            self._raise_error()
        self._seen_decl = True

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if not self._seen_decl:
            self._raise_error()

        if tag == "base" and self.base_url is None:
            href = self.get_href(attrs)
            if href is not None:
                self.base_url = href
        elif tag == "a":
            self.anchors.append(dict(attrs))

    def get_href(self, attrs: List[Tuple[str, Optional[str]]]) -> Optional[str]:
        for name, value in attrs:
            if name == "href":
                return value
        return None

    def _raise_error(self) -> None:
        raise ValueError(
            "HTML doctype missing or incorrect. Expected <!DOCTYPE html>.\n\n"
            "If you believe this error to be incorrect, try passing the "
            "command line option --use-deprecated=html5lib and please leave "
            "a comment on the pip issue at https://github.com/pypa/pip/issues/10825."
        )


def _handle_get_page_fail(
    link: Link,
    reason: Union[str, Exception],
    meth: Optional[Callable[..., None]] = None,
) -> None:
    if meth is None:
        meth = logger.debug
    meth("Could not fetch URL %s: %s - skipping", link, reason)


def _make_html_page(response: Response, cache_link_parsing: bool = True) -> HTMLPage:
    encoding = _get_encoding_from_headers(response.headers)
    return HTMLPage(
        response.content,
        encoding=encoding,
        url=response.url,
        cache_link_parsing=cache_link_parsing,
    )


def _get_html_page(
    link: Link, session: Optional[PipSession] = None
) -> Optional["HTMLPage"]:
    if session is None:
        raise TypeError(
            "_get_html_page() missing 1 required keyword argument: 'session'"
        )

    url = link.url.split("#", 1)[0]

    # Check for VCS schemes that do not support lookup as web pages.
    vcs_scheme = _match_vcs_scheme(url)
    if vcs_scheme:
        logger.warning(
            "Cannot look at %s URL %s because it does not support lookup as web pages.",
            vcs_scheme,
            link,
        )
        return None

    # Tack index.html onto file:// URLs that point to directories
    scheme, _, path, _, _, _ = urllib.parse.urlparse(url)
    if scheme == "file" and os.path.isdir(urllib.request.url2pathname(path)):
        # add trailing slash if not present so urljoin doesn't trim
        # final segment
        if not url.endswith("/"):
            url += "/"
        url = urllib.parse.urljoin(url, "index.html")
        logger.debug(" file: URL is directory, getting %s", url)

    try:
        resp = _get_html_response(url, session=session)
    except _NotHTTP:
        logger.warning(
            "Skipping page %s because it looks like an archive, and cannot "
            "be checked by a HTTP HEAD request.",
            link,
        )
    except _NotHTML as exc:
        logger.warning(
            "Skipping page %s because the %s request got Content-Type: %s."
            "The only supported Content-Type is text/html",
            link,
            exc.request_desc,
            exc.content_type,
        )
    except NetworkConnectionError as exc:
        _handle_get_page_fail(link, exc)
    except RetryError as exc:
        _handle_get_page_fail(link, exc)
    except SSLError as exc:
        reason = "There was a problem confirming the ssl certificate: "
        reason += str(exc)
        _handle_get_page_fail(link, reason, meth=logger.info)
    except requests.ConnectionError as exc:
        _handle_get_page_fail(link, f"connection error: {exc}")
    except requests.Timeout:
        _handle_get_page_fail(link, "timed out")
    else:
        return _make_html_page(resp, cache_link_parsing=link.cache_link_parsing)
    return None


class CollectedSources(NamedTuple):
    find_links: Sequence[Optional[LinkSource]]
    index_urls: Sequence[Optional[LinkSource]]


class LinkCollector:

    """
    Responsible for collecting Link objects from all configured locations,
    making network requests as needed.

    The class's main method is its collect_sources() method.
    """

    def __init__(
        self,
        session: PipSession,
        search_scope: SearchScope,
    ) -> None:
        self.search_scope = search_scope
        self.session = session

    @classmethod
    def create(
        cls,
        session: PipSession,
        options: Values,
        suppress_no_index: bool = False,
    ) -> "LinkCollector":
        """
        :param session: The Session to use to make requests.
        :param suppress_no_index: Whether to ignore the --no-index option
            when constructing the SearchScope object.
        """
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index and not suppress_no_index:
            logger.debug(
                "Ignoring indexes: %s",
                ",".join(redact_auth_from_url(url) for url in index_urls),
            )
            index_urls = []

        # Make sure find_links is a list before passing to create().
        find_links = options.find_links or []

        search_scope = SearchScope.create(
            find_links=find_links,
            index_urls=index_urls,
        )
        link_collector = LinkCollector(
            session=session,
            search_scope=search_scope,
        )
        return link_collector

    @property
    def find_links(self) -> List[str]:
        return self.search_scope.find_links

    def fetch_page(self, location: Link) -> Optional[HTMLPage]:
        """
        Fetch an HTML page containing package links.
        """
        return _get_html_page(location, session=self.session)

    def collect_sources(
        self,
        project_name: str,
        candidates_from_page: CandidatesFromPage,
    ) -> CollectedSources:
        # The OrderedDict calls deduplicate sources by URL.
        index_url_sources = collections.OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=False,
                cache_link_parsing=False,
            )
            for loc in self.search_scope.get_index_urls_locations(project_name)
        ).values()
        find_links_sources = collections.OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=True,
                cache_link_parsing=True,
            )
            for loc in self.find_links
        ).values()

        if logger.isEnabledFor(logging.DEBUG):
            lines = [
                f"* {s.link}"
                for s in itertools.chain(find_links_sources, index_url_sources)
                if s is not None and s.link is not None
            ]
            lines = [
                f"{len(lines)} location(s) to search "
                f"for versions of {project_name}:"
            ] + lines
            logger.debug("\n".join(lines))

        return CollectedSources(
            find_links=list(find_links_sources),
            index_urls=list(index_url_sources),
        )
