import asyncio
import json
import time
from pathlib import Path
from threading import RLock

import httpx
import jwt
from cachetools import TTLCache, cached
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

from backend.data.native_push_subscription import NativePushSubscriptionDTO
from backend.util.settings import Settings


async def send_apns(
    sub: NativePushSubscriptionDTO, body: str, path: str
) -> httpx.Response | None:
    config = Settings().config
    if not all((config.apns_private_key_path, config.apns_key_id, config.apns_team_id)):
        return None
    private_key = await asyncio.to_thread(Path(config.apns_private_key_path).read_text)
    token = _apns_token(private_key, config.apns_key_id, config.apns_team_id)
    host = (
        "api.sandbox.push.apple.com"
        if sub.environment == "sandbox"
        else "api.push.apple.com"
    )
    async with httpx.AsyncClient(http2=True, timeout=15) as client:
        return await client.post(
            f"https://{host}/3/device/{sub.token}",
            headers={
                "authorization": f"bearer {token}",
                "apns-topic": "com.agpt.mobile",
                "apns-push-type": "alert",
                "apns-priority": "10",
                "apns-expiration": str(int(time.time()) + 3600),
            },
            json={
                "aps": {
                    "alert": {"title": "AutoGPT", "body": body},
                    "sound": "default",
                },
                "binding_id": sub.id,
                "origin": sub.origin,
                "path": path,
            },
        )


@cached(cache=TTLCache(maxsize=4, ttl=3000), lock=RLock())
def _apns_token(private_key: str, key_id: str, team_id: str) -> str:
    return jwt.encode(
        {"iss": team_id, "iat": int(time.time())},
        private_key,
        algorithm="ES256",
        headers={"kid": key_id},
    )


@cached(cache=TTLCache(maxsize=4, ttl=3000), lock=RLock())
def _fcm_credentials(raw: str) -> tuple[str, str]:
    info = json.loads(raw)
    # Authentication always goes to Google's endpoint, including for a
    # service-account file supplied by a self-hosted deployment.
    info["token_uri"] = "https://oauth2.googleapis.com/token"
    credentials = Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/firebase.messaging"]
    )
    credentials.refresh(Request())
    if not credentials.token or not credentials.project_id:
        raise ValueError("Firebase credentials did not issue an access token")
    return credentials.project_id, credentials.token


async def send_fcm(
    sub: NativePushSubscriptionDTO, body: str, path: str
) -> httpx.Response | None:
    path_to_key = Settings().config.fcm_service_account_path
    if not path_to_key:
        return None
    raw = await asyncio.to_thread(Path(path_to_key).read_text)
    project, token = await asyncio.to_thread(_fcm_credentials, raw)
    async with httpx.AsyncClient(timeout=15) as client:
        return await client.post(
            f"https://fcm.googleapis.com/v1/projects/{project}/messages:send",
            headers={"authorization": f"Bearer {token}"},
            json={
                "message": {
                    "token": sub.token,
                    "android": {"priority": "high", "ttl": "3600s"},
                    "data": {
                        "title": "AutoGPT",
                        "body": body,
                        "binding_id": sub.id,
                        "origin": sub.origin,
                        "path": path,
                    },
                }
            },
        )
