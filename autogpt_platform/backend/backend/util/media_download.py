from backend.util.request import Requests, Response


async def download_media_url(url: str) -> Response:
    """Download a media URL, reporting an HTTP 400 response as bad input.

    HTTP 400 (Bad Request) means the URL itself is malformed or rejected by the
    server (e.g. a Wikimedia thumbnail at a non-standard width), which the caller
    must fix.  Raising ``ValueError`` makes ``Block.execute`` report it as a
    handled ``BlockExecutionError`` rather than a ``BlockUnknownError`` that
    pages Sentry.

    401 and 403 are *not* converted: a forbidden or unauthenticated URL from a
    provider-generated ``video_url`` is a platform-level problem (bad credentials,
    expired signed URL), not a user-input error.  Those responses propagate via
    ``resp.raise_for_status()`` as ``HTTPClientError`` so they still reach Sentry.

    5xx and network errors also keep their original exceptions.

    ``retry_max_attempts=1`` preserves the pre-#15340 behaviour where 429/5xx
    errors fail on the first attempt instead of retrying indefinitely with
    exponential backoff up to 300 s under ``Requests``' default retry loop.
    """
    resp = await Requests(raise_for_status=False, retry_max_attempts=1).get(url)
    if resp.status == 400:
        reason = f" ({resp.reason})" if resp.reason else ""
        raise ValueError(
            f"Could not download file from {url}: HTTP {resp.status}{reason}"
        )
    resp.raise_for_status()
    return resp
