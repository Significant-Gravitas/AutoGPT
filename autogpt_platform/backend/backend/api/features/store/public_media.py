import logging
import re
from collections.abc import Iterable
from urllib.parse import parse_qs, unquote, urlsplit

from gcloud.aio import storage as async_storage

from backend.util.settings import Settings

from . import submission_media

logger = logging.getLogger(__name__)

_GCS_URL_FORMS = (
    re.compile(
        r"https?://(?:www|storage)\.googleapis\.com/(?:download/)?storage/v1"
        r"/b/(?P<bucket>[^/?#]+)/o/(?P<path>[^?#]+)(?:[?#].*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"https?://(?:storage\.googleapis\.com|storage\.cloud\.google\.com"
        r"|commondatastorage\.googleapis\.com)/(?P<bucket>[^/?#]+)/(?P<path>[^?#]+)"
        r"(?:[?#].*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"https?://(?P<bucket>[a-z0-9._-]+)\.storage\.googleapis\.com"
        r"/(?P<path>[^?#]+)(?:[?#].*)?",
        re.IGNORECASE,
    ),
    re.compile(r"gs://(?P<bucket>[^/?#]+)/(?P<path>[^?#]+)", re.IGNORECASE),
)
_PUBLISHABLE_MEDIA_PATH = re.compile(
    r"users/(?P<owner>[^/]+)/(?P<media_type>images|videos)/(?P<filename>[^/]+)"
)


async def publish_urls(urls: Iterable[str | None], owner_id: str) -> dict[str, str]:
    settings = Settings()
    private_bucket = settings.config.resolved_private_user_data_bucket
    public_bucket = settings.config.public_site_media_bucket
    if not private_bucket or not public_bucket or private_bucket == public_bucket:
        return {}

    paths = _publishable_paths(urls, owner_id, private_bucket)
    if not paths:
        return {}

    published: dict[str, str] = {}
    async with async_storage.Storage() as async_client:
        for source_url, path in paths.items():
            try:
                await async_client.copy(
                    private_bucket, path, public_bucket, new_name=path
                )
            except Exception:
                logger.exception(f"Failed to publish {path} to {public_bucket}")
                continue
            published[source_url] = (
                f"https://storage.googleapis.com/{public_bucket}/{path}"
            )
            logger.info(f"Published {path} to {public_bucket}")
    return published


def _publishable_paths(
    urls: Iterable[str | None], owner_id: str, private_bucket: str
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for source_url in dict.fromkeys(url for url in urls if url):
        path = _object_path(source_url, private_bucket)
        if path is None:
            continue
        match = _PUBLISHABLE_MEDIA_PATH.fullmatch(path)
        if not match or match["owner"] != owner_id or not _is_valid_path(match, path):
            logger.warning(f"Not publishing {path!r}: not media owned by {owner_id}")
            continue
        paths[source_url] = path
    return paths


def _is_valid_path(match: re.Match[str], path: str) -> bool:
    try:
        canonical = submission_media.object_path(
            match["owner"], match["media_type"], match["filename"]
        )
    except ValueError:
        return False
    return canonical == path


def _object_path(url: str, bucket: str) -> str | None:
    url = url.strip()
    wrapped = urlsplit(url)
    if wrapped.path == "/_next/image":
        url = parse_qs(wrapped.query).get("url", [""])[0].strip()
    if private_path := _private_api_object_path(url):
        return private_path

    url = unquote(url)
    for form in _GCS_URL_FORMS:
        match = form.fullmatch(url)
        if match:
            return match["path"] if match["bucket"] == bucket else None
    return None


def _private_api_object_path(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return None
    path = unquote(parsed.path)
    if not path.startswith(submission_media.PRIVATE_MEDIA_PREFIX):
        return None
    parts = path.removeprefix(submission_media.PRIVATE_MEDIA_PREFIX).split("/")
    if len(parts) != 3:
        return None
    owner_id, media_type, filename = parts
    try:
        return submission_media.object_path(owner_id, media_type, filename)
    except ValueError:
        return None
