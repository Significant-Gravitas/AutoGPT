from pathlib import Path

from pydantic import SecretStr

from backend.data.model import CREDENTIALS_ADAPTER, OAuth2Credentials
from backend.integrations.codex.auth_bundle import (
    CodexAuthBundleError,
    CodexAuthBundleV1,
    decode_jwt_claims,
    decode_provider_state,
    encode_provider_state,
    read_auth_bundle,
)


def credentials_from_bundle(
    bundle: CodexAuthBundleV1,
    *,
    title: str = "ChatGPT for Codex",
) -> OAuth2Credentials:
    id_claims = decode_jwt_claims(bundle.tokens.id_token)
    access_claims = decode_jwt_claims(bundle.tokens.access_token)
    return OAuth2Credentials(
        provider="codex",
        title=title,
        username=id_claims.email,
        access_token=bundle.tokens.access_token,
        access_token_expires_at=access_claims.expires_at,
        refresh_token=bundle.tokens.refresh_token,
        scopes=[],
        refresh_strategy="provider_runtime",
        provider_state=SecretStr(encode_provider_state(bundle)),
        provider_state_version=bundle.schema_version,
        metadata={
            "plan_type": id_claims.plan_type,
            "codex_runtime_version": bundle.codex_runtime_version,
        },
    )


def bundle_from_credentials(credentials: OAuth2Credentials) -> CodexAuthBundleV1:
    if credentials.provider != "codex":
        raise CodexAuthBundleError("Credential provider is not Codex")
    if credentials.refresh_strategy != "provider_runtime":
        raise CodexAuthBundleError("Codex credential refresh ownership is invalid")
    if credentials.provider_state_version != 1 or credentials.provider_state is None:
        raise CodexAuthBundleError("Codex credential provider state is unavailable")
    return decode_provider_state(credentials.provider_state)


def checkpoint_credentials(
    credentials: OAuth2Credentials,
    auth_path: Path,
    runtime_version: str,
) -> OAuth2Credentials:
    bundle = read_auth_bundle(auth_path, runtime_version)
    return checkpoint_credentials_from_bundle(credentials, bundle)


def checkpoint_credentials_from_bundle(
    credentials: OAuth2Credentials,
    bundle: CodexAuthBundleV1,
) -> OAuth2Credentials:
    replacement = credentials_from_bundle(bundle, title=credentials.title or "")
    payload = replacement.model_dump(exclude={"id", "is_managed"})
    return OAuth2Credentials.model_validate(
        credentials.model_copy(update=payload).model_dump()
    )


def restore_credentials(payload: dict[str, object]) -> OAuth2Credentials:
    restored = CREDENTIALS_ADAPTER.validate_python(payload)
    if not isinstance(restored, OAuth2Credentials):
        raise CodexAuthBundleError("Codex credential type changed")
    return restored
