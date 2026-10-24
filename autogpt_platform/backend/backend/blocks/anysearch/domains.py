from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError
from backend.util.request import HTTPServerError

from ._api import (
    AnySearchAuth,
    AnySearchClient,
    AnySearchDomain,
    sanitize_upstream_text,
    unwrap_envelope,
)
from ._config import anysearch


class AnySearchDomainsBlock(Block):
    """Live discovery of the AnySearch domain/sub-domain catalog.

    Wraps the public GET endpoints so graphs can resolve valid ``sub_domain``
    values and their parameter contract at runtime instead of hardcoding a
    catalog that drifts with each ``topology_version``.
    """

    class Input(BlockSchemaInput):
        auth: AnySearchAuth = SchemaField(
            title="Authentication",
            description="Anonymous tier (lower rate limit, no key) or an "
            "AnySearch API key credential",
            default=AnySearchAuth.ANONYMOUS,
            advanced=False,
        )
        credentials: CredentialsMetaInput = anysearch.credentials_field(
            title="AnySearch API key",
            description="The AnySearch integration requires an API Key.",
            discriminator="auth",
            discriminator_mapping={AnySearchAuth.API_KEY.value: "anysearch"},
            credential_free_discriminator_values={AnySearchAuth.ANONYMOUS.value},
            default=None,
        )
        domains: list[AnySearchDomain] | None = SchemaField(
            description="Restrict discovery to these domains; leave empty to "
            "list the full catalog",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        domains: list[dict[str, Any]] = SchemaField(
            description="Catalog entries as returned by the API - every "
            "domain for an unrestricted call, or the selected domains with "
            "their sub-domains and parameter contracts"
        )
        topology_version: int = SchemaField(
            description="Catalog snapshot version; sub-domains evolve, so "
            "note it when recording discovered values"
        )
        request_id: str = SchemaField(description="Unique identifier for this request")

    def __init__(self):
        super().__init__(
            id="0ed822fb-d717-498e-9410-9a272f50c9c5",
            description="Discovers the AnySearch domain and sub-domain "
            "catalog (sub-domain names and their parameter contracts) "
            "via the public GET catalog endpoints",
            categories={BlockCategory.SEARCH},
            effect=BlockEffect.READ,
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": anysearch.get_test_credentials().model_dump(),
            },
            test_credentials=anysearch.get_test_credentials(),
            test_output=[
                ("domains", lambda x: isinstance(x, list) and len(x) == 1),
                ("topology_version", 9),
                ("request_id", "req-test"),
            ],
            test_mock={
                "_catalog": lambda *args, **kwargs: {
                    "code": 0,
                    "message": "success",
                    "request_id": "req-test",
                    "data": {
                        "domains": [
                            {
                                "domain": "general",
                                "description": "General web search",
                                "sub_domain_count": 1,
                            }
                        ],
                        "topology_version": 9,
                    },
                }
            },
        )

    async def _catalog(  # pragma: no cover - mocked seam, covered live
        self,
        credentials: APIKeyCredentials | None,
        domains: list[str] | None,
    ) -> dict[str, Any]:
        """GET the catalog endpoint matching the selection (mockable)."""
        client = AnySearchClient(credentials)
        if domains:
            return await client.sub_domains(domains)
        return await client.domains()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: APIKeyCredentials | None = None,
        **kwargs,
    ) -> BlockOutput:
        if input_data.auth == AnySearchAuth.ANONYMOUS:
            # Anonymous wins over a still-selected credential, the same way
            # AutoPilot honours an explicit credential-free transport.
            credentials = None
        elif credentials is None:
            raise ValueError(
                "AnySearch API key credentials are required when auth is api_key."
            )

        selected = [d.value for d in input_data.domains] if input_data.domains else None

        try:
            env = unwrap_envelope(
                await self._catalog(credentials, selected), required="domains"
            )
            raw_domains = env.data["domains"]
            if not isinstance(raw_domains, list) or not all(
                isinstance(d, dict) for d in raw_domains
            ):
                raise ValueError(
                    "malformed AnySearch catalog: data.domains is not a "
                    "list of entries"
                )
            domains = raw_domains
            topology_version = int(env.data.get("topology_version") or 0)
        except Exception as e:
            # A domain the upstream catalog retired while our enum still
            # lists it fails the whole call with a 5xx - surface that
            # drift hint alongside the upstream status.
            hint = (
                " (upstream catalog error; a selected domain may have been " "removed)"
                if isinstance(e, HTTPServerError)
                else ""
            )
            raise BlockExecutionError(
                message=f"Domains discovery failed: "
                f"{sanitize_upstream_text(str(e))}{hint}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        yield "domains", domains
        yield "topology_version", topology_version
        yield "request_id", env.request_id
