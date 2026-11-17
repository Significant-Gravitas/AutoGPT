from datetime import datetime, timedelta, timezone
from typing import Annotated, Any
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI, Security
from httpx import ASGITransport, AsyncClient
from jwt.algorithms import ECAlgorithm

from autogpt_libs.auth import config, jwt_utils


async def test_jwks_outage_is_retryable_and_does_not_cache_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = ec.generate_private_key(ec.SECP256R1())
    public = ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    public.update({"kid": "availability", "alg": "ES256", "use": "sig"})
    token = jwt.encode(
        {
            "sub": "user-1",
            "aud": "authenticated",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        private_key,
        algorithm="ES256",
        headers={"kid": "availability"},
    )
    monkeypatch.setenv("JWT_JWKS_URL", "https://auth.invalid/jwks")
    monkeypatch.setattr(config, "_settings", config.Settings())
    monkeypatch.setattr(jwt_utils, "_jwks_client", None)
    monkeypatch.setattr(jwt_utils, "_jwks_client_url", None)
    fetch = Mock(
        side_effect=[jwt.PyJWKClientConnectionError("offline"), {"keys": [public]}]
    )
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", fetch)
    app = FastAPI()

    @app.get("/me")
    async def me(
        payload: Annotated[dict[str, Any], Security(jwt_utils.get_jwt_payload)],
    ) -> dict[str, str]:
        return {"user_id": payload["sub"]}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        headers = {"Authorization": f"Bearer {token}"}
        unavailable = await client.get("/me", headers=headers)
        assert unavailable.status_code == 503
        assert unavailable.headers["Retry-After"] == "5"
        recovered = await client.get("/me", headers=headers)
        assert recovered.status_code == 200
        assert recovered.json() == {"user_id": "user-1"}
    assert fetch.call_count == 2
