import ipaddress
from functools import cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


class MCPServerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_url: str | None = None
    documentation_url: str
    setup_instructions: str
    connection_mode: Literal["hosted", "custom", "unavailable"]
    auth_mode: Literal["oauth", "token", "none", "unknown"]
    icon_id: str | None = Field(default=None, pattern=r"^[a-z0-9_-]+$")

    @field_validator("server_url", "documentation_url")
    @classmethod
    def validate_public_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or any(character.isspace() for character in value)
            or not hostname
            or "." not in hostname
            or hostname.endswith((".localhost", ".local", ".internal"))
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.port not in (None, 443)
        ):
            raise ValueError(
                "Catalog URLs must be public HTTPS URLs without credentials"
            )
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            return value
        raise ValueError("Catalog URLs must use a public hostname")

    @model_validator(mode="after")
    def validate_connection(self) -> "MCPServerMetadata":
        if self.connection_mode == "hosted":
            if not self.server_url or self.auth_mode == "unknown":
                raise ValueError("Hosted entries require a URL and documented auth")
        elif self.server_url is not None:
            raise ValueError("Custom and unavailable entries cannot prefill a URL")
        return self


class MCPCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^mcp_[a-z0-9]+(?:_[a-z0-9]+)*$")
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    official: Literal[True]
    mcp_server: MCPServerMetadata


@cache
def get_mcp_catalog() -> tuple[MCPCatalogEntry, ...]:
    content = Path(__file__).with_suffix(".json").read_text(encoding="utf-8")
    return parse_mcp_catalog(content)


def parse_mcp_catalog(content: str) -> tuple[MCPCatalogEntry, ...]:
    entries = TypeAdapter(list[MCPCatalogEntry]).validate_json(content)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("MCP catalog provider names must be unique")
    return tuple(entries)
