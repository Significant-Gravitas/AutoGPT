import pytest

from backend.cli.oauth_tool import format_sql_insert, generate_app_credentials


def test_generate_public_app_with_explicit_client_id() -> None:
    credentials = generate_app_credentials(
        name="AutoGPT Local Executor",
        description="Local PC shim",
        client_id="autogpt-local-executor",
        is_public=True,
        redirect_uris=["http://localhost:41899/callback"],
        scopes=["USE_TOOLS"],
    )

    assert credentials["client_id"] == "autogpt-local-executor"
    assert credentials["is_public"] is True

    sql = format_sql_insert(credentials)
    assert '"isPublic"' in sql
    assert "autogpt-local-executor" in sql
    assert "does not receive or use a client secret" in sql
    assert "true\n);" in sql
    assert "ARRAY{" not in sql
    assert "'{\"http://localhost:41899/callback\"}'::TEXT[]" in sql
    assert '\'{"USE_TOOLS"}\'::"APIKeyPermission"[]' in sql


def test_generate_app_rejects_unsafe_explicit_client_id() -> None:
    with pytest.raises(ValueError, match="client_id"):
        generate_app_credentials(
            name="unsafe",
            client_id="bad' id",
            redirect_uris=["http://localhost/callback"],
            scopes=["USE_TOOLS"],
        )


@pytest.mark.parametrize(
    "redirect_uri",
    [
        "javascript:alert(document.domain)",
        "data:text/html,<script>alert(1)</script>",
        "file:///tmp/callback",
        "http://example.com/callback",
        "https://user:password@example.com/callback",
        "https://example.com/callback#fragment",
        "https://example.com/with space",
        "https://example.com\\@evil.test/callback",
    ],
)
def test_generate_app_rejects_unsafe_redirect_uri(redirect_uri: str) -> None:
    with pytest.raises(ValueError, match="Redirect URIs"):
        generate_app_credentials(
            name="unsafe",
            redirect_uris=[redirect_uri],
            scopes=["USE_TOOLS"],
        )


@pytest.mark.parametrize(
    "redirect_uri",
    [
        "https://example.com/callback",
        "http://localhost:41899/callback",
        "http://127.0.0.1:41899/callback",
        "http://[::1]:41899/callback",
    ],
)
def test_generate_app_accepts_safe_redirect_uri(redirect_uri: str) -> None:
    credentials = generate_app_credentials(
        name="safe",
        redirect_uris=[redirect_uri],
        scopes=["USE_TOOLS"],
    )

    assert credentials["redirect_uris"] == [redirect_uri]
