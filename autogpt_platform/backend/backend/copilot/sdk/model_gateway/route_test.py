import pytest
from pydantic import SecretStr, ValidationError

from backend.copilot.sdk.model_gateway.route import (
    ApiKeyCredential,
    CodexLeaseCredential,
    GatewayRoute,
)
from backend.copilot.sdk.model_gateway.testing.codex_lease import (
    codex_route,
    fake_access_token,
    fake_codex_lease,
)


def test_a_local_route_keeps_its_url_without_the_trailing_slash():
    route = GatewayRoute(
        adapter="openai_chat", model="m", base_url="http://127.0.0.1:8080/v1/"
    )
    assert route.upstream_base == "http://127.0.0.1:8080/v1"


@pytest.mark.parametrize("adapter", ["anthropic_passthrough", "openai_chat"])
def test_a_local_route_needs_an_upstream_url(adapter):
    with pytest.raises(ValidationError, match="needs an upstream base URL"):
        GatewayRoute(adapter=adapter, model="m")


@pytest.mark.parametrize(
    "url, problem",
    [
        ("ftp://127.0.0.1:8080", "http"),
        ("http://user:pass@127.0.0.1:8080", "credential"),
        ("http://127.0.0.1:8080/v1?key=abc", "query"),
        ("http:///v1", "host"),
    ],
)
def test_a_local_route_refuses_urls_that_could_leak_or_misroute(url, problem):
    with pytest.raises(ValidationError, match=problem):
        GatewayRoute(adapter="anthropic_passthrough", model="m", base_url=url)


def test_the_codex_endpoint_cannot_be_overridden_by_a_route():
    with pytest.raises(ValidationError, match="Codex endpoint is fixed"):
        GatewayRoute(
            adapter="codex_responses",
            model="gpt",
            base_url="https://elsewhere.example",
            credential=CodexLeaseCredential(lease=fake_codex_lease()),
        )


def test_a_codex_route_needs_the_account_lease():
    with pytest.raises(ValidationError, match="credential lease"):
        GatewayRoute(
            adapter="codex_responses",
            model="gpt",
            credential=ApiKeyCredential(api_key=SecretStr("sk-x")),
        )


def test_a_chatgpt_credential_never_serves_a_local_route():
    with pytest.raises(ValidationError, match="only serves the Codex route"):
        GatewayRoute(
            adapter="openai_chat",
            model="m",
            base_url="http://127.0.0.1:1/v1",
            credential=CodexLeaseCredential(lease=fake_codex_lease()),
        )


def test_the_route_exposes_its_secrets_for_redaction_but_never_prints_them():
    route = GatewayRoute(
        adapter="anthropic_passthrough",
        model="m",
        base_url="http://127.0.0.1:1",
        credential=ApiKeyCredential(api_key=SecretStr("sk-upstream-key-123")),
    )
    assert route.api_key() == "sk-upstream-key-123"
    assert route.secret_values() == ("sk-upstream-key-123",)
    assert "sk-upstream-key-123" not in repr(route)
    assert "sk-upstream-key-123" not in str(route.model_dump())


def test_a_codex_route_redacts_the_access_token_it_sends():
    route = codex_route("gpt-test")
    assert route.secret_values() == (fake_access_token(),)
    assert route.api_key() is None


def test_a_route_without_a_key_has_nothing_to_redact():
    route = GatewayRoute(
        adapter="openai_chat", model="m", base_url="http://127.0.0.1:1/v1"
    )
    assert route.secret_values() == ()
    assert route.api_key() is None
