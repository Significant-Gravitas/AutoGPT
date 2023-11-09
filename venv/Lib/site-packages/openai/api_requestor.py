import asyncio
import json
import platform
import sys
import threading
import warnings
from contextlib import asynccontextmanager
from json import JSONDecodeError
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Union,
    overload,
)
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
import requests

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import openai
from openai import error, util, version
from openai.openai_response import OpenAIResponse
from openai.util import ApiType

TIMEOUT_SECS = 600
MAX_CONNECTION_RETRIES = 2

# Has one attribute per thread, 'session'.
_thread_context = threading.local()


def _build_api_url(url, query):
    scheme, netloc, path, base_query, fragment = urlsplit(url)

    if base_query:
        query = "%s&%s" % (base_query, query)

    return urlunsplit((scheme, netloc, path, query, fragment))


def _requests_proxies_arg(proxy) -> Optional[Dict[str, str]]:
    """Returns a value suitable for the 'proxies' argument to 'requests.request."""
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return {"http": proxy, "https": proxy}
    elif isinstance(proxy, dict):
        return proxy.copy()
    else:
        raise ValueError(
            "'openai.proxy' must be specified as either a string URL or a dict with string URL under the https and/or http keys."
        )


def _aiohttp_proxies_arg(proxy) -> Optional[str]:
    """Returns a value suitable for the 'proxies' argument to 'aiohttp.ClientSession.request."""
    if proxy is None:
        return None
    elif isinstance(proxy, str):
        return proxy
    elif isinstance(proxy, dict):
        return proxy["https"] if "https" in proxy else proxy["http"]
    else:
        raise ValueError(
            "'openai.proxy' must be specified as either a string URL or a dict with string URL under the https and/or http keys."
        )


def _make_session() -> requests.Session:
    if not openai.verify_ssl_certs:
        warnings.warn("verify_ssl_certs is ignored; openai always verifies.")
    s = requests.Session()
    proxies = _requests_proxies_arg(openai.proxy)
    if proxies:
        s.proxies = proxies
    s.mount(
        "https://",
        requests.adapters.HTTPAdapter(max_retries=MAX_CONNECTION_RETRIES),
    )
    return s


def parse_stream_helper(line: bytes) -> Optional[str]:
    if line:
        if line.strip() == b"data: [DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        if line.startswith(b"data: "):
            line = line[len(b"data: "):]
            return line.decode("utf-8")
        else:
            return None
    return None


def parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


async def parse_stream_async(rbody: aiohttp.StreamReader):
    async for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


class APIRequestor:
    def __init__(
        self,
        key=None,
        api_base=None,
        api_type=None,
        api_version=None,
        organization=None,
    ):
        self.api_base = api_base or openai.api_base
        self.api_key = key or util.default_api_key()
        self.api_type = (
            ApiType.from_str(api_type)
            if api_type
            else ApiType.from_str(openai.api_type)
        )
        self.api_version = api_version or openai.api_version
        self.organization = organization or openai.organization

    @classmethod
    def format_app_info(cls, info):
        str = info["name"]
        if info["version"]:
            str += "/%s" % (info["version"],)
        if info["url"]:
            str += " (%s)" % (info["url"],)
        return str

    @overload
    def request(
        self,
        method,
        url,
        params,
        headers,
        files,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Iterator[OpenAIResponse], bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        *,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Iterator[OpenAIResponse], bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: Literal[False] = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[OpenAIResponse, bool, str]:
        pass

    @overload
    def request(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: bool = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        pass

    def request(
        self,
        method,
        url,
        params=None,
        headers=None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
        result = self.request_raw(
            method.lower(),
            url,
            params=params,
            supplied_headers=headers,
            files=files,
            stream=stream,
            request_id=request_id,
            request_timeout=request_timeout,
        )
        resp, got_stream = self._interpret_response(result, stream)
        return resp, got_stream, self.api_key

    @overload
    async def arequest(
        self,
        method,
        url,
        params,
        headers,
        files,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[AsyncGenerator[OpenAIResponse, None], bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        *,
        stream: Literal[True],
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[AsyncGenerator[OpenAIResponse, None], bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: Literal[False] = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[OpenAIResponse, bool, str]:
        pass

    @overload
    async def arequest(
        self,
        method,
        url,
        params=...,
        headers=...,
        files=...,
        stream: bool = ...,
        request_id: Optional[str] = ...,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = ...,
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool, str]:
        pass

    async def arequest(
        self,
        method,
        url,
        params=None,
        headers=None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool, str]:
        ctx = aiohttp_session()
        session = await ctx.__aenter__()
        try:
            result = await self.arequest_raw(
                method.lower(),
                url,
                session,
                params=params,
                supplied_headers=headers,
                files=files,
                request_id=request_id,
                request_timeout=request_timeout,
            )
            resp, got_stream = await self._interpret_async_response(result, stream)
        except Exception:
            await ctx.__aexit__(None, None, None)
            raise
        if got_stream:

            async def wrap_resp():
                assert isinstance(resp, AsyncGenerator)
                try:
                    async for r in resp:
                        yield r
                finally:
                    await ctx.__aexit__(None, None, None)

            return wrap_resp(), got_stream, self.api_key
        else:
            await ctx.__aexit__(None, None, None)
            return resp, got_stream, self.api_key

    def handle_error_response(self, rbody, rcode, resp, rheaders, stream_error=False):
        try:
            error_data = resp["error"]
        except (KeyError, TypeError):
            raise error.APIError(
                "Invalid response object from API: %r (HTTP response code "
                "was %d)" % (rbody, rcode),
                rbody,
                rcode,
                resp,
            )

        if "internal_message" in error_data:
            error_data["message"] += "\n\n" + error_data["internal_message"]

        util.log_info(
            "OpenAI API error received",
            error_code=error_data.get("code"),
            error_type=error_data.get("type"),
            error_message=error_data.get("message"),
            error_param=error_data.get("param"),
            stream_error=stream_error,
        )

        # Rate limits were previously coded as 400's with code 'rate_limit'
        if rcode == 429:
            return error.RateLimitError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode in [400, 404, 415]:
            return error.InvalidRequestError(
                error_data.get("message"),
                error_data.get("param"),
                error_data.get("code"),
                rbody,
                rcode,
                resp,
                rheaders,
            )
        elif rcode == 401:
            return error.AuthenticationError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode == 403:
            return error.PermissionError(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif rcode == 409:
            return error.TryAgain(
                error_data.get("message"), rbody, rcode, resp, rheaders
            )
        elif stream_error:
            # TODO: we will soon attach status codes to stream errors
            parts = [error_data.get("message"), "(Error occurred while streaming.)"]
            message = " ".join([p for p in parts if p is not None])
            return error.APIError(message, rbody, rcode, resp, rheaders)
        else:
            return error.APIError(
                f"{error_data.get('message')} {rbody} {rcode} {resp} {rheaders}",
                rbody,
                rcode,
                resp,
                rheaders,
            )

    def request_headers(
        self, method: str, extra, request_id: Optional[str]
    ) -> Dict[str, str]:
        user_agent = "OpenAI/v1 PythonBindings/%s" % (version.VERSION,)
        if openai.app_info:
            user_agent += " " + self.format_app_info(openai.app_info)

        uname_without_node = " ".join(
            v for k, v in platform.uname()._asdict().items() if k != "node"
        )
        ua = {
            "bindings_version": version.VERSION,
            "httplib": "requests",
            "lang": "python",
            "lang_version": platform.python_version(),
            "platform": platform.platform(),
            "publisher": "openai",
            "uname": uname_without_node,
        }
        if openai.app_info:
            ua["application"] = openai.app_info

        headers = {
            "X-OpenAI-Client-User-Agent": json.dumps(ua),
            "User-Agent": user_agent,
        }

        headers.update(util.api_key_to_header(self.api_type, self.api_key))

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        if self.api_version is not None and self.api_type == ApiType.OPEN_AI:
            headers["OpenAI-Version"] = self.api_version
        if request_id is not None:
            headers["X-Request-Id"] = request_id
        if openai.debug:
            headers["OpenAI-Debug"] = "true"
        headers.update(extra)

        return headers

    def _validate_headers(
        self, supplied_headers: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if supplied_headers is None:
            return headers

        if not isinstance(supplied_headers, dict):
            raise TypeError("Headers must be a dictionary")

        for k, v in supplied_headers.items():
            if not isinstance(k, str):
                raise TypeError("Header keys must be strings")
            if not isinstance(v, str):
                raise TypeError("Header values must be strings")
            headers[k] = v

        # NOTE: It is possible to do more validation of the headers, but a request could always
        # be made to the API manually with invalid headers, so we need to handle them server side.

        return headers

    def _prepare_request_raw(
        self,
        url,
        supplied_headers,
        method,
        params,
        files,
        request_id: Optional[str],
    ) -> Tuple[str, Dict[str, str], Optional[bytes]]:
        abs_url = "%s%s" % (self.api_base, url)
        headers = self._validate_headers(supplied_headers)

        data = None
        if method == "get" or method == "delete":
            if params:
                encoded_params = urlencode(
                    [(k, v) for k, v in params.items() if v is not None]
                )
                abs_url = _build_api_url(abs_url, encoded_params)
        elif method in {"post", "put"}:
            if params and files:
                data = params
            if params and not files:
                data = json.dumps(params).encode()
                headers["Content-Type"] = "application/json"
        else:
            raise error.APIConnectionError(
                "Unrecognized HTTP method %r. This may indicate a bug in the "
                "OpenAI bindings. Please contact support@openai.com for "
                "assistance." % (method,)
            )

        headers = self.request_headers(method, headers, request_id)

        util.log_debug("Request to OpenAI API", method=method, path=abs_url)
        util.log_debug("Post details", data=data, api_version=self.api_version)

        return abs_url, headers, data

    def request_raw(
        self,
        method,
        url,
        *,
        params=None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files=None,
        stream: bool = False,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> requests.Response:
        abs_url, headers, data = self._prepare_request_raw(
            url, supplied_headers, method, params, files, request_id
        )

        if not hasattr(_thread_context, "session"):
            _thread_context.session = _make_session()
        try:
            result = _thread_context.session.request(
                method,
                abs_url,
                headers=headers,
                data=data,
                files=files,
                stream=stream,
                timeout=request_timeout if request_timeout else TIMEOUT_SECS,
            )
        except requests.exceptions.Timeout as e:
            raise error.Timeout("Request timed out: {}".format(e)) from e
        except requests.exceptions.RequestException as e:
            raise error.APIConnectionError(
                "Error communicating with OpenAI: {}".format(e)
            ) from e
        util.log_debug(
            "OpenAI API response",
            path=abs_url,
            response_code=result.status_code,
            processing_ms=result.headers.get("OpenAI-Processing-Ms"),
            request_id=result.headers.get("X-Request-Id"),
        )
        # Don't read the whole stream for debug logging unless necessary.
        if openai.log == "debug":
            util.log_debug(
                "API response body", body=result.content, headers=result.headers
            )
        return result

    async def arequest_raw(
        self,
        method,
        url,
        session,
        *,
        params=None,
        supplied_headers: Optional[Dict[str, str]] = None,
        files=None,
        request_id: Optional[str] = None,
        request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> aiohttp.ClientResponse:
        abs_url, headers, data = self._prepare_request_raw(
            url, supplied_headers, method, params, files, request_id
        )

        if isinstance(request_timeout, tuple):
            timeout = aiohttp.ClientTimeout(
                connect=request_timeout[0],
                total=request_timeout[1],
            )
        else:
            timeout = aiohttp.ClientTimeout(
                total=request_timeout if request_timeout else TIMEOUT_SECS
            )

        if files:
            # TODO: Use `aiohttp.MultipartWriter` to create the multipart form data here.
            # For now we use the private `requests` method that is known to have worked so far.
            data, content_type = requests.models.RequestEncodingMixin._encode_files(  # type: ignore
                files, data
            )
            headers["Content-Type"] = content_type
        request_kwargs = {
            "method": method,
            "url": abs_url,
            "headers": headers,
            "data": data,
            "proxy": _aiohttp_proxies_arg(openai.proxy),
            "timeout": timeout,
        }
        try:
            result = await session.request(**request_kwargs)
            util.log_info(
                "OpenAI API response",
                path=abs_url,
                response_code=result.status,
                processing_ms=result.headers.get("OpenAI-Processing-Ms"),
                request_id=result.headers.get("X-Request-Id"),
            )
            # Don't read the whole stream for debug logging unless necessary.
            if openai.log == "debug":
                util.log_debug(
                    "API response body", body=result.content, headers=result.headers
                )
            return result
        except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            raise error.Timeout("Request timed out") from e
        except aiohttp.ClientError as e:
            raise error.APIConnectionError("Error communicating with OpenAI") from e

    def _interpret_response(
        self, result: requests.Response, stream: bool
    ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool]:
        """Returns the response(s) and a bool indicating whether it is a stream."""
        if stream and "text/event-stream" in result.headers.get("Content-Type", ""):
            return (
                self._interpret_response_line(
                    line, result.status_code, result.headers, stream=True
                )
                for line in parse_stream(result.iter_lines())
            ), True
        else:
            return (
                self._interpret_response_line(
                    result.content.decode("utf-8"),
                    result.status_code,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    async def _interpret_async_response(
        self, result: aiohttp.ClientResponse, stream: bool
    ) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool]:
        """Returns the response(s) and a bool indicating whether it is a stream."""
        if stream and "text/event-stream" in result.headers.get("Content-Type", ""):
            return (
                self._interpret_response_line(
                    line, result.status, result.headers, stream=True
                )
                async for line in parse_stream_async(result.content)
            ), True
        else:
            try:
                await result.read()
            except aiohttp.ClientError as e:
                util.log_warn(e, body=result.content)
            return (
                self._interpret_response_line(
                    (await result.read()).decode("utf-8"),
                    result.status,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    def _interpret_response_line(
        self, rbody: str, rcode: int, rheaders, stream: bool
    ) -> OpenAIResponse:
        # HTTP 204 response code does not have any content in the body.
        if rcode == 204:
            return OpenAIResponse(None, rheaders)

        if rcode == 503:
            raise error.ServiceUnavailableError(
                "The server is overloaded or not ready yet.",
                rbody,
                rcode,
                headers=rheaders,
            )
        try:
            if 'text/plain' in rheaders.get('Content-Type'):
                data = rbody
            else:
                data = json.loads(rbody)
        except (JSONDecodeError, UnicodeDecodeError) as e:
            raise error.APIError(
                f"HTTP code {rcode} from API ({rbody})", rbody, rcode, headers=rheaders
            ) from e
        resp = OpenAIResponse(data, rheaders)
        # In the future, we might add a "status" parameter to errors
        # to better handle the "error while streaming" case.
        stream_error = stream and "error" in resp.data
        if stream_error or not 200 <= rcode < 300:
            raise self.handle_error_response(
                rbody, rcode, resp.data, rheaders, stream_error=stream_error
            )
        return resp


@asynccontextmanager
async def aiohttp_session() -> AsyncIterator[aiohttp.ClientSession]:
    user_set_session = openai.aiosession.get()
    if user_set_session:
        yield user_set_session
    else:
        async with aiohttp.ClientSession() as session:
            yield session
