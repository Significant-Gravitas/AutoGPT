import base64
import json
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from backend.integrations.codex.auth_bundle import (
    CodexAuthBundleError,
    auth_bundle_fingerprint,
    materialize_auth_bundle,
    read_auth_bundle,
)
from backend.integrations.codex.credential_codec import (
    bundle_from_credentials,
    credentials_from_bundle,
    restore_credentials,
)
from backend.util.encryption import JSONCryptor


def _jwt(payload: dict[str, object]) -> str:
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )
    return f"header.{encoded}.signature"


def _write_auth(path: Path, marker: str) -> tuple[str, str, str]:
    id_token = _jwt(
        {
            "email": f"{marker}@example.com",
            "https://api.openai.com/auth": {"chatgpt_plan_type": "pro"},
        }
    )
    access_token = _jwt({"exp": 2_000_000_000})
    refresh_token = f"refresh-{marker}"
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": id_token,
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "account_id": f"account-{marker}",
                },
                "last_refresh": None,
            }
        ),
        encoding="utf-8",
    )
    return id_token, access_token, refresh_token


def test_bundle_round_trips_through_oauth_credentials(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    tokens = _write_auth(auth_path, "alpha")

    bundle = read_auth_bundle(auth_path, "0.144.4")
    credentials = credentials_from_bundle(bundle)

    assert credentials.provider == "codex"
    assert credentials.refresh_strategy == "provider_runtime"
    assert credentials.username == "alpha@example.com"
    assert credentials.access_token == SecretStr(tokens[1])
    assert bundle_from_credentials(credentials) == bundle
    assert all(token not in repr(credentials) for token in tokens)


def test_provider_state_round_trips_through_existing_encryption(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    tokens = _write_auth(auth_path, "alpha")
    credentials = credentials_from_bundle(read_auth_bundle(auth_path, "0.144.4"))
    cryptor = JSONCryptor(Fernet.generate_key().decode())

    encrypted = cryptor.encrypt(credentials.model_dump())
    restored = restore_credentials(cryptor.decrypt(encrypted))

    assert all(token not in encrypted for token in tokens)
    assert restored.id == credentials.id
    assert bundle_from_credentials(restored) == bundle_from_credentials(credentials)


def test_materialized_bundles_do_not_cross_users(tmp_path: Path):
    source_a = tmp_path / "source-a.json"
    source_b = tmp_path / "source-b.json"
    tokens_a = _write_auth(source_a, "alpha")
    tokens_b = _write_auth(source_b, "bravo")
    bundle_a = read_auth_bundle(source_a, "0.144.4")
    bundle_b = read_auth_bundle(source_b, "0.144.4")
    target_a = tmp_path / "a" / "auth.json"
    target_b = tmp_path / "b" / "auth.json"

    materialize_auth_bundle(bundle_a, target_a)
    materialize_auth_bundle(bundle_b, target_b)

    assert auth_bundle_fingerprint(
        read_auth_bundle(target_a, "0.144.4")
    ) == auth_bundle_fingerprint(bundle_a)
    assert auth_bundle_fingerprint(
        read_auth_bundle(target_b, "0.144.4")
    ) == auth_bundle_fingerprint(bundle_b)
    assert tokens_a[2] not in target_b.read_text(encoding="utf-8")
    assert tokens_b[2] not in target_a.read_text(encoding="utf-8")


def test_auth_bundle_rejects_api_key_and_unknown_fields(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps({"auth_mode": "apiKey", "OPENAI_API_KEY": "secret"}),
        encoding="utf-8",
    )

    with pytest.raises(CodexAuthBundleError):
        read_auth_bundle(auth_path, "0.144.4")
