import base64
import binascii
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)


class CodexAuthBundleError(ValueError):
    pass


class CodexAuthTokensV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id_token: SecretStr
    access_token: SecretStr
    refresh_token: SecretStr
    account_id: str | None = None

    @field_validator("id_token", "access_token", "refresh_token")
    @classmethod
    def reject_empty_tokens(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("required token is empty")
        return value


class CodexAuthBundleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    auth_mode: Literal["chatgpt"] = "chatgpt"
    tokens: CodexAuthTokensV1
    last_refresh: datetime | None = None
    codex_runtime_version: str


class CodexJwtClaims(BaseModel):
    email: str | None = None
    plan_type: str | None = None
    chatgpt_user_id: str | None = None
    chatgpt_account_id: str | None = None
    expires_at: int | None = None


class _CodexAuthFileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    auth_mode: Literal["chatgpt"]
    openai_api_key: None = Field(default=None, alias="OPENAI_API_KEY")
    tokens: CodexAuthTokensV1
    last_refresh: datetime | None = None
    agent_identity: None = None
    personal_access_token: None = None
    bedrock_api_key: None = None


class _JwtProfileClaims(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: str | None = None


class _JwtAuthClaims(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chatgpt_plan_type: str | None = None
    chatgpt_user_id: str | None = None
    user_id: str | None = None
    chatgpt_account_id: str | None = None


class _JwtPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    email: str | None = None
    expires_at: int | None = Field(default=None, alias="exp")
    profile: _JwtProfileClaims | None = Field(
        default=None,
        alias="https://api.openai.com/profile",
    )
    auth: _JwtAuthClaims | None = Field(
        default=None,
        alias="https://api.openai.com/auth",
    )


def read_auth_bundle(auth_path: Path, runtime_version: str) -> CodexAuthBundleV1:
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
        auth_file = _CodexAuthFileV1.model_validate(payload)
    except (OSError, ValueError, ValidationError):
        raise CodexAuthBundleError(
            f"Codex auth file {auth_path.name!r} failed strict ChatGPT validation"
        ) from None

    return CodexAuthBundleV1(
        tokens=auth_file.tokens,
        last_refresh=auth_file.last_refresh,
        codex_runtime_version=runtime_version,
    )


def encode_provider_state(bundle: CodexAuthBundleV1) -> str:
    return json.dumps(_bundle_payload(bundle), separators=(",", ":"), sort_keys=True)


def decode_provider_state(provider_state: SecretStr) -> CodexAuthBundleV1:
    try:
        payload = json.loads(provider_state.get_secret_value())
        return CodexAuthBundleV1.model_validate(payload)
    except (binascii.Error, UnicodeDecodeError, ValueError, ValidationError):
        raise CodexAuthBundleError(
            "Encrypted Codex provider state is invalid"
        ) from None


def materialize_auth_bundle(bundle: CodexAuthBundleV1, auth_path: Path) -> None:
    auth_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": _tokens_payload(bundle.tokens),
        "last_refresh": _datetime_payload(bundle.last_refresh),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    descriptor = os.open(auth_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as auth_file:
        auth_file.write(encoded)
        auth_file.flush()
        os.fsync(auth_file.fileno())


def auth_bundle_fingerprint(bundle: CodexAuthBundleV1) -> str:
    return hashlib.sha256(encode_provider_state(bundle).encode()).hexdigest()


def decode_jwt_claims(token: SecretStr) -> CodexJwtClaims:
    try:
        parts = token.get_secret_value().split(".")
        if len(parts) != 3 or not all(parts):
            raise ValueError
        padding = "=" * (-len(parts[1]) % 4)
        payload = _JwtPayload.model_validate_json(
            base64.urlsafe_b64decode(parts[1] + padding)
        )
    except (ValueError, ValidationError):
        raise CodexAuthBundleError("Codex token has an invalid JWT payload") from None

    auth = payload.auth or _JwtAuthClaims()
    profile = payload.profile or _JwtProfileClaims()
    return CodexJwtClaims(
        email=payload.email or profile.email,
        plan_type=auth.chatgpt_plan_type,
        chatgpt_user_id=auth.chatgpt_user_id or auth.user_id,
        chatgpt_account_id=auth.chatgpt_account_id,
        expires_at=payload.expires_at,
    )


def _bundle_payload(bundle: CodexAuthBundleV1) -> dict[str, object]:
    return {
        "schema_version": bundle.schema_version,
        "auth_mode": bundle.auth_mode,
        "tokens": _tokens_payload(bundle.tokens),
        "last_refresh": _datetime_payload(bundle.last_refresh),
        "codex_runtime_version": bundle.codex_runtime_version,
    }


def _tokens_payload(tokens: CodexAuthTokensV1) -> dict[str, str | None]:
    return {
        "id_token": tokens.id_token.get_secret_value(),
        "access_token": tokens.access_token.get_secret_value(),
        "refresh_token": tokens.refresh_token.get_secret_value(),
        "account_id": tokens.account_id,
    }


def _datetime_payload(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None
