"""An MCP client must get from the MCP server's resource metadata to the
provider's endpoints using nothing but the URLs the documents give it."""

from urllib.parse import urlparse

import httpx
from fastapi.routing import APIRoute
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
)
from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata
from starlette.routing import Match

import backend.api.features.oauth
from backend.api.external.v2.mcp_server import WELL_KNOWN_PROTECTED_RESOURCE_PATH
from backend.api.rest_api import app


async def test_the_resource_metadata_leads_to_the_authorization_server_metadata():
    """Discovery as the MCP SDK does it, against the real app: the issuer must
    equal the authorization server the resource names (RFC 8414 §3.3), and
    neither document may carry a null member."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resource_response = await client.get(WELL_KNOWN_PROTECTED_RESOURCE_PATH)
        assert resource_response.status_code == 200, resource_response.text
        resource = ProtectedResourceMetadata.model_validate(resource_response.json())
        issuer = str(resource.authorization_servers[0])

        discovery_url = build_oauth_authorization_server_metadata_discovery_urls(
            issuer, str(resource.resource)
        )[0]
        response = await client.get(urlparse(discovery_url).path)

    assert response.status_code == 200, response.text
    assert None not in [*resource_response.json().values(), *response.json().values()]
    metadata = OAuthMetadata.model_validate(response.json())
    assert str(metadata.issuer) == issuer
    assert _route_for("POST", urlparse(str(metadata.token_endpoint)).path) is (
        backend.api.features.oauth.token
    )


def _route_for(method: str, path: str):
    scope = {"type": "http", "method": method, "path": path, "headers": []}
    for route in app.routes:
        if isinstance(route, APIRoute) and route.matches(scope)[0] == Match.FULL:
            return route.endpoint
    return None
