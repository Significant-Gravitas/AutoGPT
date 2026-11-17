"""The external API's own addresses, as the host app mounts it."""

import re
from urllib.parse import urljoin

import fastapi
import fastapi.testclient
import pytest

from backend.api.external.fastapi_app import external_api
from backend.api.external.v1.app import DESCRIPTION as V1_DESCRIPTION


@pytest.fixture
def client() -> fastapi.testclient.TestClient:
    host = fastapi.FastAPI()
    host.mount("/external-api", external_api)
    return fastapi.testclient.TestClient(host, follow_redirects=False)


@pytest.mark.parametrize("prefix", ["", "/_agpt"])
def test_the_root_redirects_to_v2s_docs_behind_any_proxy_prefix(
    client: fastapi.testclient.TestClient, prefix: str
) -> None:
    response = client.get("/external-api/")

    page = f"https://example.com{prefix}/external-api/"
    assert urljoin(page, response.headers["location"]) == (
        f"https://example.com{prefix}/api/v2/docs"
    )


@pytest.mark.parametrize("prefix", ["", "/_agpt"])
def test_v1s_docs_link_to_v2s_docs_behind_any_proxy_prefix(prefix: str) -> None:
    link = re.search(r"\[v2\]\(([^)]+)\)", V1_DESCRIPTION)

    assert link
    page = f"https://example.com{prefix}/external-api/v1/docs"
    assert urljoin(page, link.group(1)) == f"https://example.com{prefix}/api/v2/docs"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/versions.json"])
def test_only_the_versions_publish_a_spec_and_docs(
    client: fastapi.testclient.TestClient, path: str
) -> None:
    """A spec of the wrapper app documents nothing a client can call."""
    assert client.get(f"/external-api{path}").status_code == 404


def test_v1s_spec_is_still_found_at_its_old_address(
    client: fastapi.testclient.TestClient,
) -> None:
    """Published docs and generated clients fetch it from there."""
    response = client.get("/external-api/openapi.json")

    assert response.status_code == 308
    assert response.headers["location"] == "/external-api/v1/openapi.json"
