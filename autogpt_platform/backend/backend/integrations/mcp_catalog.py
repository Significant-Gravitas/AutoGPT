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


class MCPServerURLPreset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    url: str


class MCPServerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_url: str | None = None
    documentation_url: str
    setup_instructions: str
    connection_mode: Literal["hosted", "custom"]
    auth_methods: list[Literal["oauth", "bearer", "basic", "none"]] = Field(
        min_length=1
    )
    server_url_options: list[MCPServerURLPreset] = Field(default_factory=list)
    oauth_server_url: str | None = None
    oauth_scopes: list[str] | None = None
    oauth_write_scopes: list[str] = Field(default_factory=list)
    icon_id: str | None = Field(default=None, pattern=r"^[a-z0-9_-]+$")
    provider: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
        description="Block provider this server belongs to, when the platform "
        "also ships blocks for it. Lets the UI file an MCP connection under the "
        "same service as its API-key credential instead of a generic MCP group.",
    )

    @field_validator("server_url", "documentation_url", "oauth_server_url")
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
            if not self.server_url:
                raise ValueError("Hosted entries require a URL")
        elif self.server_url is not None:
            raise ValueError("Custom entries cannot prefill a URL")
        if len(set(self.auth_methods)) != len(self.auth_methods):
            raise ValueError("Authentication methods must be unique")
        if self.oauth_write_scopes and self.oauth_scopes is None:
            raise ValueError("Optional write scopes require explicit default scopes")
        if set(self.oauth_scopes or []) & set(self.oauth_write_scopes):
            raise ValueError("Default and optional write scopes must be disjoint")
        if (
            self.oauth_server_url
            or self.oauth_scopes is not None
            or self.oauth_write_scopes
        ) and "oauth" not in self.auth_methods:
            raise ValueError("OAuth settings require OAuth authentication")
        for option in self.server_url_options:
            self.validate_public_url(option.url)
        return self

    @field_validator("oauth_scopes", "oauth_write_scopes")
    @classmethod
    def validate_scope_tokens(cls, scopes: list[str] | None) -> list[str] | None:
        if scopes is None:
            return None
        if len(set(scopes)) != len(scopes):
            raise ValueError("OAuth scopes must be unique")
        for scope in scopes:
            if not scope or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in scope
            ):
                raise ValueError("OAuth scopes must be nonempty tokens")
        return scopes


class MCPCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^mcp_[a-z0-9]+(?:_[a-z0-9]+)*$")
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    mcp_server: MCPServerMetadata


@cache
def get_mcp_catalog() -> tuple[MCPCatalogEntry, ...]:
    content = Path(__file__).with_suffix(".json").read_text(encoding="utf-8")
    return parse_mcp_catalog(content)


def _catalog_url_key(url: str) -> tuple[str, str] | None:
    try:
        parsed = urlsplit(url.strip())
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.port not in (None, 443)
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            return None
        return parsed.hostname, parsed.path.rstrip("/")
    except ValueError:
        return None


def get_mcp_catalog_entry_for_url(server_url: str) -> MCPCatalogEntry | None:
    key = _catalog_url_key(server_url)
    if key is None:
        return None
    for entry in get_mcp_catalog():
        server = entry.mcp_server
        urls = [
            server.server_url,
            server.oauth_server_url,
            *(option.url for option in server.server_url_options),
        ]
        if any(url and _catalog_url_key(url) == key for url in urls):
            return entry
    return None


def parse_mcp_catalog(content: str) -> tuple[MCPCatalogEntry, ...]:
    entries = TypeAdapter(list[MCPCatalogEntry]).validate_json(content)
    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("MCP catalog provider names must be unique")
    return tuple(entries)


MCPToolEffect = Literal["read", "external", "irreversible"]


class MCPEffectMap(BaseModel):
    """A catalogued server's tools by effect, written from its published list.

    A tool the map does not name has no known effect, so it asks on first use.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    read: list[str] = Field(default_factory=list)
    external: list[str] = Field(default_factory=list)
    # External writes that cannot be taken back.
    irreversible: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_disjoint(self) -> "MCPEffectMap":
        tools = [*self.read, *self.external, *self.irreversible]
        if len(set(tools)) != len(tools):
            raise ValueError("A tool appears in more than one effect list")
        return self

    def effect_of(self, tool: str) -> MCPToolEffect | None:
        if tool in self.irreversible:
            return "irreversible"
        if tool in self.external:
            return "external"
        return "read" if tool in self.read else None


def mcp_tool_effect(server_url: str, tool: str) -> MCPToolEffect | None:
    """None: the server is not catalogued, or its map does not name the tool."""
    entry = get_mcp_catalog_entry_for_url(server_url)
    effects = get_mcp_effect_maps().get(entry.name) if entry else None
    return effects.effect_of(tool) if effects else None


@cache
def get_mcp_effect_maps() -> dict[str, MCPEffectMap]:
    content = Path(__file__).with_name("mcp_effects.json").read_text(encoding="utf-8")
    return parse_mcp_effect_maps(content)


def parse_mcp_effect_maps(content: str) -> dict[str, MCPEffectMap]:
    maps = TypeAdapter(dict[str, MCPEffectMap]).validate_json(content)
    unknown = set(maps) - {entry.name for entry in get_mcp_catalog()}
    if unknown:
        raise ValueError(f"Effect maps for servers not in the catalog: {unknown}")
    return maps
