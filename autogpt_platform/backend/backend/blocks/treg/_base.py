"""Shared base class and schemas for Treg blocks."""

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from backend.data.model import NodeExecutionStats
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import HttpMethod, TregError, TregResponse, error_message, send
from ._config import treg

# Routed calls stop spending past this unless the block says otherwise.
# Treg's own default is $1 per call; a typical lookup costs a few cents.
DEFAULT_MAX_COST_USD = 0.5
MAX_COST_CEILING_USD = 25.0


class TregInput(BlockSchemaInput):
    credentials: CredentialsMetaInput = treg.credentials_field(
        description=(
            "Your Treg API key: a team token or dashboard API key from "
            "treg.to. Calls are paid from that team's Treg balance."
        )
    )


class TregRoutedInput(TregInput):
    max_cost_usd: float = SchemaField(
        default=DEFAULT_MAX_COST_USD,
        ge=0,
        le=MAX_COST_CEILING_USD,
        advanced=True,
        description=(
            "The most this run may spend, in USD, across every provider Treg "
            "tries. A call that would go over it is refused without charge."
        ),
    )
    try_more_providers: bool = SchemaField(
        default=True,
        advanced=True,
        description=(
            "When a provider has no answer, try the next cheapest one, up to the "
            "spending cap. Some providers bill a miss. Turn off to stop after "
            "the first provider."
        ),
    )


class TregRoutedOutput(BlockSchemaOutput):
    found: bool = SchemaField(
        description="True when a provider had an answer, false when none did."
    )
    served_by: str = SchemaField(
        description="The Treg tool that answered, e.g. 'hunter.email.finder'. "
        "Emitted only when one did."
    )
    providers_tried: list[str] = SchemaField(
        description="Providers Treg asked, in order."
    )
    cost_usd: float = SchemaField(description="What Treg charged for this run, in USD.")
    call_id: str = SchemaField(
        description="Treg's id for this call, to look it up in your Treg activity."
    )
    advice: str = SchemaField(
        description="Treg's note on how far to trust the answer, e.g. that an "
        "email was found but not checked. Emitted only when there is one."
    )
    raw: dict[str, Any] = SchemaField(
        description="The serving provider's full answer, with fields beyond the "
        "standard ones. Emitted only when one answered."
    )


class TregAttempt(BaseModel):
    endpoint_id: str | None = None
    provider: str | None = None
    outcome: str | None = None


class TregRouteMeta(BaseModel):
    served_by: str | None = None
    outcome: str | None = None
    tried: list[TregAttempt] = Field(default_factory=list)
    advice: str | None = None
    verification: dict[str, Any] | None = None
    ignored_filters: list[str] = Field(default_factory=list)


class TregRouted(BaseModel):
    """A routed answer: the standard output fields plus Treg's notes."""

    output: dict[str, Any] = Field(default_factory=dict)
    raw: Any = None
    meta: TregRouteMeta = Field(default_factory=TregRouteMeta, alias="_treg")
    cost_usd: float = 0.0
    call_id: str | None = None

    def get(self, field: str) -> Any:
        return self.output.get(field)


class TregBlockBase(Block):
    """Sends Treg calls and records what each one charged."""

    @staticmethod
    async def call_treg(
        credentials: APIKeyCredentials,
        method: HttpMethod,
        path: str,
        params: dict[str, Any] | None = None,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> TregResponse:
        return await send(
            credentials, method, path, params=params, body=body, headers=headers
        )

    async def request(
        self,
        credentials: APIKeyCredentials,
        method: HttpMethod,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> TregResponse:
        """Send one call, record its charge, and raise if it failed.

        The charge is recorded before any error is raised, so a call Treg
        billed is billed on the platform too, whatever happened next.
        """
        response = await self.call_treg(
            credentials, method, path, params, body, headers
        )
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=response.cost_usd, provider_cost_type="cost_usd"
            )
        )
        if not response.ok:
            raise TregError(error_message(response))
        return response

    async def route(
        self,
        credentials: APIKeyCredentials,
        capability: str,
        body: dict[str, Any],
        input_data: TregRoutedInput,
        headers: dict[str, str] | None = None,
    ) -> TregRouted:
        """Call a routed tool, ``treg.<capability>``, and parse its answer."""
        response = await self.request(
            credentials,
            "POST",
            f"call/treg.{capability}",
            body={k: v for k, v in body.items() if v not in (None, "", [], {})},
            headers={**route_headers(input_data), **(headers or {})},
        )
        if response.status == 202:
            raise TregError(
                "The provider is still working on this call after Treg's 60 second "
                f"wait (call id {response.call_id}). It may still finish and charge; "
                "check your Treg activity before running it again."
            )
        if not isinstance(response.body, dict):
            raise TregError(f"Treg returned an unexpected answer for treg.{capability}")
        routed = TregRouted.model_validate(response.body)
        routed.cost_usd = response.cost_usd
        routed.call_id = response.call_id
        return routed

    @staticmethod
    def route_details(routed: TregRouted, found: bool) -> Iterator[tuple[str, Any]]:
        """The outputs every routed block shares, after its own fields."""
        yield "found", found
        if found and routed.meta.served_by:
            yield "served_by", routed.meta.served_by
        providers = [a.provider for a in routed.meta.tried if a.provider]
        yield "providers_tried", providers
        yield "cost_usd", routed.cost_usd
        if routed.call_id:
            yield "call_id", routed.call_id
        if routed.meta.advice:
            yield "advice", routed.meta.advice
        if found and isinstance(routed.raw, dict):
            yield "raw", routed.raw


def rows(routed: TregRouted, field: str) -> list[dict[str, Any]]:
    """The answer's list of records, keeping only well-formed ones."""
    return [row for row in routed.get(field) or [] if isinstance(row, dict)]


def route_headers(input_data: TregRoutedInput) -> dict[str, str]:
    headers = {"X-Treg-Route-Max-Cost": f"{input_data.max_cost_usd:g}"}
    if not input_data.try_more_providers:
        headers["X-Treg-Route-Waterfall"] = "0"
    return headers


def require_one(identities: dict[str, bool], hint: str) -> None:
    """Fail before spending when none of the ways to name the subject is given."""
    if not any(identities.values()):
        raise TregError(f"Nothing to look up: {hint}")


def typed_fields(
    output: dict[str, Any], spec: dict[str, type]
) -> Iterator[tuple[str, Any]]:
    """Yield the output fields that are present and of the declared type.

    Providers fill Treg's standard fields when they have them; a field that is
    missing, empty or of another type is left out rather than failing the run.
    """
    for name, kind in spec.items():
        value = coerce(output.get(name), kind)
        if value is not None:
            yield name, value


def coerce(value: Any, kind: type) -> Any:
    if value is None or value == "":
        return None
    if kind is bool:
        return value if isinstance(value, bool) else None
    if kind is int:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).replace(",", "").strip()
        return int(text) if text.isdigit() else None
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
    if kind is str:
        return value if isinstance(value, str) else None
    if kind is list:
        return value if isinstance(value, list) else None
    return value
