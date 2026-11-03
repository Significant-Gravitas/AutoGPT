"""
OAuth authentication for the shim.

Uses AutoGPT's existing OAuth 2.0 provider with:
- Authorization Code + PKCE flow
- Localhost callback server on port 41899
- Token storage in OS keychain via the `keyring` library
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import keyring

from .config import ShimConfig, _validate_platform_endpoint

KEYCHAIN_SERVICE = "autogpt-local-executor"
KEYCHAIN_ACCESS_TOKEN_KEY = "access_token"
KEYCHAIN_REFRESH_TOKEN_KEY = "refresh_token"
# How long `autogpt-shim auth` waits for the browser to come back.
CALLBACK_TIMEOUT_SECONDS = 300.0


class CredentialScopeError(ValueError):
    """Stored credentials cannot be used with the configured destinations."""


def _canonical_endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(
        _validate_platform_endpoint(value, {"https", "http", "wss", "ws"})
    )
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if parsed.scheme in {"https", "wss"} else 80
    if parsed.port is not None and parsed.port != default_port:
        host += f":{parsed.port}"
    return parsed._replace(netloc=host).geturl()


@dataclass(frozen=True)
class CredentialScope:
    backend_url: str
    authorize_url: str
    token_url: str
    revoke_url: str
    websocket_url: str
    client_id: str

    @classmethod
    def from_config(cls, config: ShimConfig) -> CredentialScope:
        return cls(
            backend_url=_canonical_endpoint(config._platform_base()),
            authorize_url=_canonical_endpoint(config.derived_oauth_url.rstrip("/") + "/authorize"),
            token_url=_canonical_endpoint(config.derived_oauth_token_url),
            revoke_url=_canonical_endpoint(config.derived_oauth_revoke_url),
            websocket_url=_canonical_endpoint(config.derived_ws_url.rstrip("/")),
            client_id=config.oauth_client_id,
        )

    def ensure_matches(self, config: ShimConfig) -> None:
        if self != self.from_config(config):
            raise CredentialScopeError(
                "OAuth configuration has changed. Restart and run `autogpt-shim auth` "
                "with the intended deployment configuration before connecting."
            )


class KeychainTokenStore:
    """
    Stores OAuth tokens in the OS keychain using the `keyring` library.

    Backends by platform:
        macOS   → Keychain Services
        Linux   → Secret Service (GNOME Keyring / KWallet)
        Windows → Windows Credential Manager
    """

    def __init__(self, config: ShimConfig) -> None:
        self.scope = CredentialScope.from_config(config)
        identity = json.dumps(asdict(self.scope), sort_keys=True, separators=(",", ":"))
        self._record_key = "oauth_v1_" + hashlib.sha256(identity.encode()).hexdigest()

    async def get_access_token(self) -> str | None:
        record = await asyncio.to_thread(self._read_record)
        return record["access_token"] if record is not None else None

    async def store_tokens(self, access_token: str, refresh_token: str) -> None:
        """Persist one versioned, deployment-bound token pair atomically."""
        if not isinstance(access_token, str) or not access_token:
            raise CredentialScopeError("OAuth response did not contain a valid access token.")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise CredentialScopeError("OAuth response did not contain a valid refresh token.")
        record = json.dumps(
            {
                "version": 1,
                "scope": asdict(self.scope),
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
            separators=(",", ":"),
        )
        await asyncio.to_thread(keyring.set_password, KEYCHAIN_SERVICE, self._record_key, record)

    def clear_tokens(self) -> None:
        """Remove only this deployment's record, leaving legacy credentials untouched."""
        try:
            keyring.delete_password(KEYCHAIN_SERVICE, self._record_key)
        except keyring.errors.PasswordDeleteError:
            pass

    async def get_refresh_token(self) -> str | None:
        record = await asyncio.to_thread(self._read_record)
        return record["refresh_token"] if record is not None else None

    def _read_record(self) -> dict[str, str] | None:
        stored = keyring.get_password(KEYCHAIN_SERVICE, self._record_key)
        if stored is None:
            if any(
                keyring.get_password(KEYCHAIN_SERVICE, key)
                for key in (KEYCHAIN_ACCESS_TOKEN_KEY, KEYCHAIN_REFRESH_TOKEN_KEY)
            ):
                raise CredentialScopeError(
                    "Unscoped legacy credentials cannot be reused. Run `autogpt-shim auth` "
                    "with the intended deployment configuration. Legacy credentials were "
                    "left untouched; revoke them at their original deployment."
                )
            return None
        try:
            record = json.loads(stored)
            if (
                not isinstance(record, dict)
                or record.get("version") != 1
                or record.get("scope") != asdict(self.scope)
                or not isinstance(record.get("access_token"), str)
                or not record["access_token"]
                or not isinstance(record.get("refresh_token"), str)
                or not record["refresh_token"]
            ):
                raise ValueError
        except (ValueError, TypeError):
            raise CredentialScopeError(
                "Stored credentials could not be verified for this deployment. "
                "Run `autogpt-shim auth` with the intended deployment configuration."
            ) from None
        return {"access_token": record["access_token"], "refresh_token": record["refresh_token"]}


class OAuthFlow:
    """
    Runs the Authorization Code + PKCE flow against the AutoGPT OAuth provider.
    """

    # Only what a Local PC connection needs; USE_TOOLS also covers web search
    # and fetch through the External API.
    SCOPES = ["CONNECT_LOCAL_EXECUTOR"]

    def __init__(self, config: ShimConfig, token_store: KeychainTokenStore) -> None:
        self.config = config
        self.token_store = token_store
        self._scope = CredentialScope.from_config(config)
        self._ensure_scope()

    def _ensure_scope(self) -> None:
        self._scope.ensure_matches(self.config)
        if (
            isinstance(self.token_store, KeychainTokenStore)
            and self.token_store.scope != self._scope
        ):
            raise CredentialScopeError(
                "Stored credentials belong to different OAuth configuration. "
                "Run `autogpt-shim auth` with the intended deployment configuration."
            )

    def run(self) -> None:
        """Execute the full auth flow interactively."""
        code_verifier, code_challenge = self._generate_pkce_pair()
        state = secrets.token_urlsafe(32)
        server, auth_code, oauth_error, callback_port = self._bind_callback_server(
            expected_state=state
        )
        auth_url = self._build_auth_url(
            code_challenge,
            state=state,
            redirect_port=callback_port,
        )

        print("\nOpening browser for AutoGPT authentication...")
        print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")
        webbrowser.open(auth_url)

        code = self._wait_for_callback(server, auth_code, oauth_error)
        tokens = asyncio.run(self._exchange_code(code, code_verifier, redirect_port=callback_port))
        self._ensure_scope()
        asyncio.run(self.token_store.store_tokens(tokens["access_token"], tokens["refresh_token"]))
        print("Authentication successful. Tokens stored in OS keychain.")

    async def refresh_token(self) -> None:
        """Use the stored refresh token to obtain a new access token."""
        self._ensure_scope()
        refresh = await self.token_store.get_refresh_token()
        if not refresh:
            raise ValueError("No refresh token stored. Run `autogpt-shim auth` first.")
        async with httpx.AsyncClient() as client:
            self._ensure_scope()
            resp = await client.post(
                self._scope.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh,
                    "client_id": self._scope.client_id,
                },
            )
            resp.raise_for_status()
            tokens = resp.json()
        self._ensure_scope()
        await self.token_store.store_tokens(
            tokens["access_token"],
            tokens.get("refresh_token", refresh),
        )

    async def revoke_tokens(self) -> None:
        """Revoke both token types remotely before deleting local credentials.

        The desktop shim is a public client, so it sends the well-known client
        id with an empty secret. Any HTTP or keychain read error aborts before
        local deletion, leaving the credentials available for a retry.
        """
        self._ensure_scope()
        access = await self.token_store.get_access_token()
        refresh = await self.token_store.get_refresh_token()
        tokens = (
            (access, "access_token"),
            (refresh, "refresh_token"),
        )
        async with httpx.AsyncClient() as client:
            for token, token_type_hint in tokens:
                if not token:
                    continue
                self._ensure_scope()
                resp = await client.post(
                    self._scope.revoke_url,
                    json={
                        "token": token,
                        "token_type_hint": token_type_hint,
                        "client_id": self._scope.client_id,
                        "client_secret": "",
                    },
                )
                resp.raise_for_status()
        self._ensure_scope()
        self.token_store.clear_tokens()

    @staticmethod
    def _generate_pkce_pair() -> tuple[str, str]:
        code_verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return code_verifier, code_challenge

    def _build_auth_url(
        self,
        code_challenge: str,
        *,
        state: str | None = None,
        redirect_port: int | None = None,
    ) -> str:
        self._ensure_scope()
        port = redirect_port or self.config.oauth_redirect_port
        params = {
            "response_type": "code",
            "client_id": self._scope.client_id,
            "redirect_uri": f"http://localhost:{port}/callback",
            "scope": " ".join(self.SCOPES),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state or secrets.token_urlsafe(32),
        }
        return self._scope.authorize_url + "?" + urllib.parse.urlencode(params)

    def _bind_callback_server(
        self, *, expected_state: str
    ) -> tuple[HTTPServer, list[str], list[str], int]:
        auth_code: list[str] = []
        oauth_error: list[str] = []

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                state = params.get("state", [""])[0]
                if parsed.path != "/callback":
                    self.send_response(404)
                    message = b"Not found"
                elif not secrets.compare_digest(state, expected_state):
                    oauth_error.append("OAuth callback state did not match")
                    self.send_response(400)
                    message = b"Authentication failed: invalid state."
                elif "error" in params:
                    oauth_error.append(params["error"][0])
                    self.send_response(400)
                    message = b"Authentication was denied or failed."
                elif "code" in params and params["code"][0]:
                    auth_code.append(params["code"][0])
                    self.send_response(200)
                    message = b"<h1>Authenticated! You can close this tab.</h1>"
                else:
                    oauth_error.append("No authorization code received")
                    self.send_response(400)
                    message = b"Authentication failed: no authorization code."
                self.end_headers()
                self.wfile.write(message)

            def log_message(self, *args):
                pass

        first_port = self.config.oauth_redirect_port
        for port in range(first_port, first_port + 12):
            try:
                server = HTTPServer(("127.0.0.1", port), CallbackHandler)
                return server, auth_code, oauth_error, port
            except OSError:
                continue
        raise RuntimeError(f"Could not bind OAuth callback on ports {first_port}-{first_port + 11}")

    @staticmethod
    def _wait_for_callback(
        server: HTTPServer,
        auth_code: list[str],
        oauth_error: list[str],
        *,
        timeout: float = CALLBACK_TIMEOUT_SECONDS,
    ) -> str:
        """Serve requests until the redirect to /callback arrives or `timeout`
        passes. A request for another path, such as a browser fetching
        /favicon.ico, does not end the wait."""
        deadline = time.monotonic() + timeout
        try:
            while not auth_code and not oauth_error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                server.timeout = remaining
                server.handle_request()
        finally:
            server.server_close()

        if not auth_code:
            detail = oauth_error[0] if oauth_error else "OAuth callback timed out"
            raise ValueError(detail)
        return auth_code[0]

    async def _exchange_code(
        self,
        auth_code: str,
        code_verifier: str,
        *,
        redirect_port: int | None = None,
    ) -> dict:
        self._ensure_scope()
        port = redirect_port or self.config.oauth_redirect_port
        async with httpx.AsyncClient() as client:
            self._ensure_scope()
            resp = await client.post(
                self._scope.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "redirect_uri": f"http://localhost:{port}/callback",
                    "client_id": self._scope.client_id,
                    "code_verifier": code_verifier,
                },
            )
            resp.raise_for_status()
            return resp.json()
