"""Operator-selected upstream, model and in-memory credential for one CLI launch."""

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.models import CodexReasoningEffort
from backend.integrations.codex.transport import CodexCredentialLease
from backend.integrations.credential_lease import CredentialLease

AdapterKind = Literal["anthropic_passthrough", "openai_chat", "codex_responses"]


class NoCredential(BaseModel):
    """An upstream that takes no key, such as a llama.cpp server on loopback."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["none"] = "none"


class ApiKeyCredential(BaseModel):
    """An operator-configured key, held in memory for the gateway's lifetime."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["api_key"] = "api_key"
    api_key: SecretStr


class CodexLeaseCredential(BaseModel):
    """The connected ChatGPT account, as leased for this turn."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["codex_lease"] = "codex_lease"
    lease: CodexCredentialLease | CredentialLease


GatewayCredential = Annotated[
    NoCredential | ApiKeyCredential | CodexLeaseCredential,
    Field(discriminator="kind"),
]


class GatewayRoute(BaseModel):
    """The route's URL, credential, model and actual context window.

    Pass-through takes an Anthropic root; Chat takes an OpenAI API base.
    The context window fills gaps in overflow errors; the CLI gets its window
    through environment variables. Effort comes from the platform route.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    adapter: AdapterKind
    model: str = Field(min_length=1)
    base_url: str | None = None
    credential: GatewayCredential = Field(default_factory=NoCredential)
    context_window: int | None = Field(default=None, gt=0)
    effort: CodexReasoningEffort | None = None
    timeout_seconds: float = Field(default=1800.0, gt=0)

    @model_validator(mode="after")
    def _check_upstream(self) -> "GatewayRoute":
        codex_credential = self.credential.kind == "codex_lease"
        if self.credential.kind == "api_key":
            key = self.credential.api_key.get_secret_value()
            if 0 < len(key) < 4:
                raise ValueError("An operator key must have at least four characters")
        if self.adapter == "codex_responses":
            if self.base_url is not None:
                raise ValueError("The Codex endpoint is fixed; a route cannot set it")
            if not codex_credential:
                raise ValueError("A Codex route needs the account's credential lease")
            return self
        if codex_credential:
            raise ValueError("A ChatGPT credential only serves the Codex route")
        if self.base_url is None:
            raise ValueError(f"The {self.adapter} route needs an upstream base URL")
        _check_base_url(self.base_url)
        return self

    @property
    def upstream_base(self) -> str:
        """The configured base URL without a trailing slash."""
        if self.base_url is None:
            raise RuntimeError("This route has no configurable upstream URL")
        return self.base_url.rstrip("/")

    def api_key(self) -> str | None:
        """The operator's key for this upstream, if it takes one."""
        credential = self.credential
        if credential.kind == "api_key":
            return credential.api_key.get_secret_value()
        return None

    def secret_values(self) -> tuple[str, ...]:
        """Every credential string this route sends upstream, for redaction."""
        credential = self.credential
        if credential.kind == "api_key":
            return (credential.api_key.get_secret_value(),)
        if credential.kind == "codex_lease":
            leased = credential.lease.credentials
            if isinstance(leased, OAuth2Credentials):
                return (leased.access_token.get_secret_value(),)
        return ()


def _check_base_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("The upstream base URL must be an http(s) URL with a host")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Put the upstream key in the credential, not in the URL")
    if parts.query or parts.fragment:
        raise ValueError("The upstream base URL cannot carry a query or fragment")
