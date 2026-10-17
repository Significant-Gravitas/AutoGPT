"""Provider, models and REST calls for Google's Developer Knowledge API.

The API (https://developers.google.com/knowledge) searches and serves Google's
public developer documentation. It takes a Google Cloud API key, sent in the
``X-Goog-Api-Key`` header so the key never ends up in a URL.
"""

from typing import Any, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr

from backend.sdk import APIKeyCredentials, CredentialsMetaInput, ProviderBuilder
from backend.util.exceptions import BlockError, BlockExecutionError, BlockInputError
from backend.util.json import loads
from backend.util.request import Requests, Response

API_ROOT = "https://developerknowledge.googleapis.com"
MAX_DOCUMENTS = 20
CORPUS_URL = "https://developers.google.com/knowledge/reference/corpus-reference"

developer_docs = (
    ProviderBuilder("google_developer_docs")
    .with_description("Search and read Google's developer documentation")
    .with_api_key("GOOGLE_DEVELOPER_DOCS_API_KEY", "Google Developer Knowledge API key")
    .build()
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="87fd0122-d577-4d10-ad26-f0817c971069",
    provider="google_developer_docs",
    api_key=SecretStr("mock-google-developer-knowledge-api-key"),
    title="Mock Google Developer Knowledge API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def developer_docs_credentials_field() -> CredentialsMetaInput:
    return developer_docs.credentials_field(
        description=(
            "A Google Cloud API key with the Developer Knowledge API enabled "
            "(Google Cloud console > APIs & Services > Credentials)"
        )
    )


class DeveloperDocPassage(BaseModel):
    """A passage from a page of Google's developer documentation."""

    document_name: str = Field(
        description=(
            "The page's document name (documents/...). Pass it to Get Google "
            "Developer Docs for the whole page"
        )
    )
    title: Optional[str] = Field(default=None, description="The page title")
    url: Optional[str] = Field(default=None, description="Link to the page")
    content: str = Field(default="", description="The passage, in Markdown")
    relevance_score: Optional[float] = Field(
        default=None, description="How relevant the passage is, from 0 to 1"
    )
    site: Optional[str] = Field(
        default=None, description="The documentation site, e.g. firebase.google.com"
    )
    updated_at: Optional[str] = Field(
        default=None, description="When the page last changed (RFC 3339)"
    )


class DeveloperDoc(BaseModel):
    """A whole page of Google's developer documentation."""

    name: str = Field(description="The document name (documents/...)")
    title: Optional[str] = Field(default=None, description="The page title")
    url: Optional[str] = Field(default=None, description="Link to the page")
    description: Optional[str] = Field(
        default=None, description="A short description of the page"
    )
    content: str = Field(default="", description="The whole page, in Markdown")
    site: Optional[str] = Field(
        default=None, description="The documentation site, e.g. firebase.google.com"
    )
    updated_at: Optional[str] = Field(
        default=None, description="When the page last changed (RFC 3339)"
    )


class DeveloperKnowledgeError(Exception):
    """A non-2xx reply from the Developer Knowledge API."""

    def __init__(self, status: int, message: str, reason: str = ""):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.reason = reason


async def call_developer_knowledge(
    method: str,
    path: str,
    api_key: str,
    *,
    params: dict[str, str] | list[tuple[str, str]] | None = None,
    body: dict[str, Any] | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Call a Developer Knowledge REST method and return the JSON reply.

    Throttled and 5xx replies are retried up to ``max_attempts`` times in all.
    """
    response = await Requests(
        trusted_origins=[API_ROOT],
        raise_for_status=False,
        retry_max_attempts=max_attempts,
    ).request(
        method,
        f"{API_ROOT}/{path}",
        headers={"X-Goog-Api-Key": api_key},
        params=params,
        json=body,
    )
    if not response.ok:
        raise api_error(response)
    reply = loads(response.content, fallback=None) if response.content else {}
    return reply if isinstance(reply, dict) else {}


def api_error(response: Response) -> DeveloperKnowledgeError:
    """Read Google's error envelope: ``{"error": {"message", "details"}}``."""
    reply = loads(response.content, fallback=None) if response.content else None
    error = reply.get("error") if isinstance(reply, dict) else None
    error = error if isinstance(error, dict) else {}
    reasons = [
        detail["reason"]
        for detail in error.get("details", [])
        if isinstance(detail, dict) and detail.get("reason")
    ]
    return DeveloperKnowledgeError(
        response.status,
        error.get("message") or response.reason or "no details",
        reasons[0] if reasons else "",
    )


def developer_docs_error(
    exc: DeveloperKnowledgeError,
    block_name: str,
    block_id: str,
    quota_hint: str = "",
) -> BlockError:
    """Turn a Developer Knowledge API error into a message the user can act on."""
    if exc.status == 401 or exc.reason == "API_KEY_INVALID":
        message = (
            "Google rejected the API key. Check the Google Developer Docs "
            "credential, or create a new key in the Google Cloud console under "
            "APIs & Services > Credentials."
        )
    elif exc.reason == "API_KEY_SERVICE_BLOCKED":
        message = (
            "This API key isn't allowed to call the Developer Knowledge API. In "
            "the Google Cloud console, edit the key and add Developer Knowledge "
            "API to its API restrictions."
        )
    elif exc.status == 400:
        return BlockInputError(
            message=f"Google's developer docs rejected the request: {exc.message}",
            block_name=block_name,
            block_id=block_id,
        )
    elif exc.status == 404:
        message = (
            f"Google's developer docs have no such page ({exc.message}). Use "
            "document names from Search Google Developer Docs, or links to pages "
            f"on the sites Google indexes ({CORPUS_URL})."
        )
    elif exc.status == 429:
        message = (
            "The Developer Knowledge API quota of this key's Google Cloud project "
            f"is used up: {exc.message}"
        )
        if quota_hint:
            message = f"{message} {quota_hint}"
    else:
        message = f"Google Developer Knowledge API error {exc.status}: {exc.message}"
    return BlockExecutionError(
        message=message, block_name=block_name, block_id=block_id
    )


def build_filter(sites: list[str], custom_filter: str) -> str:
    """Combine the sites to search and a raw filter into one AIP-160 filter."""
    hosts = dict.fromkeys(host for host in map(site_host, sites) if host)
    clauses = []
    if hosts:
        clauses.append(" OR ".join(f'data_source = "{host}"' for host in hosts))
    if custom_filter.strip():
        clauses.append(custom_filter.strip())
    if len(clauses) < 2:
        return "".join(clauses)
    return " AND ".join(f"({clause})" for clause in clauses)


def site_host(value: str) -> str:
    """Accept a site as a host name or a URL: both give ``firebase.google.com``."""
    value = value.strip().lower()
    if not value:
        return ""
    return urlsplit(value if "://" in value else f"https://{value}").netloc


def to_document_name(value: str) -> str:
    """Accept a document name (``documents/...``) or a link to the page."""
    value = value.strip()
    if value.startswith("documents/"):
        return value
    parts = urlsplit(value if "://" in value else f"https://{value}")
    return f"documents/{parts.netloc.lower()}{parts.path}"


def to_passage(chunk: dict[str, Any]) -> DeveloperDocPassage:
    """Map an API ``DocumentChunk`` to a passage."""
    document = chunk.get("document") or {}
    return DeveloperDocPassage(
        document_name=chunk.get("parent") or document.get("name") or "",
        title=document.get("title"),
        url=document.get("uri"),
        content=chunk.get("content") or "",
        relevance_score=chunk.get("relevanceScore"),
        site=document.get("dataSource"),
        updated_at=document.get("updateTime"),
    )


def to_developer_doc(document: dict[str, Any]) -> DeveloperDoc:
    """Map an API ``Document`` to a page."""
    return DeveloperDoc(
        name=document.get("name") or "",
        title=document.get("title"),
        url=document.get("uri"),
        description=document.get("description"),
        content=document.get("content") or "",
        site=document.get("dataSource"),
        updated_at=document.get("updateTime"),
    )


def document_names(passages: list[DeveloperDocPassage]) -> list[str]:
    """The pages behind a list of passages, once each, in rank order."""
    return list(dict.fromkeys(p.document_name for p in passages if p.document_name))
