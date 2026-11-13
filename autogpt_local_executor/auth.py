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
import errno
import hashlib
import io
import json
import secrets
import selectors
import socket
import sys
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import keyring

from .config import ShimConfig, _validate_platform_endpoint

KEYCHAIN_SERVICE = "autogpt-local-executor"
# How long `autogpt-shim auth` waits for the browser to come back.
CALLBACK_TIMEOUT_SECONDS = 300.0
# The redirect ports the OAuth app registers (41899-41910).
CALLBACK_FIRST_PORT = 41899
CALLBACK_LAST_PORT = 41910
# How long the callback server waits for one connection's request. It takes
# one connection at a time, so without this a connection that never sends
# (a browser preconnect, say) would hold off the real redirect.
CALLBACK_READ_TIMEOUT_SECONDS = 10.0
# How long one request may take to arrive in full. The timeout above bounds
# each read, so a client sending a byte every few seconds would otherwise
# hold the server, and keep the sign-in waiting past its deadline.
CALLBACK_REQUEST_SECONDS = 20.0


class _DeadlineReader(io.RawIOBase):
    """A socket's incoming bytes, read with a deadline for all of them: each
    read waits at most `per_read` seconds and never past `total` seconds from
    the start, then raises TimeoutError, which http.server answers by
    dropping the connection."""

    def __init__(self, sock: socket.socket, *, per_read: float, total: float) -> None:
        super().__init__()
        self._sock = sock
        self._per_read = per_read
        self._deadline = time.monotonic() + total

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the request took too long to arrive")
        self._sock.settimeout(min(self._per_read, remaining))
        return self._sock.recv_into(buffer)


class _CallbackServer(HTTPServer):
    """The OAuth callback's server. On Windows it takes its port
    exclusively: there SO_REUSEADDR, which HTTPServer sets, binds a port
    another socket is already listening on, so the fallback to the next
    port never ran and the redirect could reach the other listener.
    Elsewhere SO_REUSEADDR stays; it only lets a new sign-in bind over the
    last one's closed connections.

    The redirect URI says `localhost`, which a browser may try as ::1 before
    127.0.0.1, so where the machine has IPv6 the server has a companion on
    ::1 and the same port, closed with it."""

    companion: _CallbackServer | None = None
    # Never SO_REUSEPORT, which Python 3.14's first betas had HTTPServer set:
    # another program of the same user could listen on the port too and take
    # the redirect.
    allow_reuse_port = False

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def server_close(self) -> None:
        super().server_close()
        if self.companion is not None:
            self.companion.server_close()


class _CallbackServer6(_CallbackServer):
    address_family = socket.AF_INET6


# What binding ::1 fails with on a machine without IPv6 loopback.
_NO_IPV6_LOOPBACK = frozenset(
    {errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, getattr(errno, "EPROTONOSUPPORT", errno.EAFNOSUPPORT)}
)


class CredentialScopeError(ValueError):
    """Stored credentials cannot be used with the configured destinations."""


class OAuthDenied(ValueError):
    """The sign-in redirect came back with an error instead of a code. Its
    text is the provider's error and reason, kept to printable text: safe to
    show, unlike other failures, whose text can hold a token."""


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


@dataclass(frozen=True)
class RefreshedTokens:
    """The pair a refresh exchange returned, and the refresh token it used up."""

    used_refresh_token: str
    access_token: str
    refresh_token: str


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
        # A refreshed pair not stored yet (OAuthFlow.refresh_token), kept for
        # every session of this process: they share the store.
        self.unsaved_refresh: RefreshedTokens | None = None

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
        """Remove only this deployment's record."""
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


# The longest reason from a denied sign-in the terminal shows.
_OAUTH_ERROR_CHARS = 300


def _oauth_error_text(params: dict[str, list[str]]) -> str:
    """A denied sign-in's error code and, when the redirect gives one, its
    reason ("Local PC isn't available for this account yet."), kept to
    printable text, since the terminal shows it."""
    error = params["error"][0]
    reason = (params.get("error_description") or [""])[0]
    text = f"{error}: {reason}" if reason else error
    return "".join(char for char in text if char.isprintable())[:_OAUTH_ERROR_CHARS]


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
        """Use the stored refresh token to obtain a new access token.

        The platform rotates the refresh token on every exchange, so once one
        goes through the token the keychain holds is used up, and sending it
        again ends the sign-in. A new pair the keychain then fails to store
        (locked meanwhile, say) is kept on the token store until a later
        refresh stores it, without exchanging again, for as long as the
        keychain still holds the token that exchange used up: a sign-in
        stored since (`autogpt-shim auth`) goes ahead of it. A restart loses
        it, so the daemon retries in the same process."""
        self._ensure_scope()
        refresh = await self.token_store.get_refresh_token()
        tokens = self.token_store.unsaved_refresh
        if tokens is not None and tokens.used_refresh_token != refresh:
            tokens = self.token_store.unsaved_refresh = None
        if tokens is None:
            if not refresh:
                raise ValueError("No refresh token stored. Run `autogpt-shim auth` first.")
            tokens = self.token_store.unsaved_refresh = await self._exchange_refresh_token(refresh)
        self._ensure_scope()
        await self.token_store.store_tokens(tokens.access_token, tokens.refresh_token)
        self.token_store.unsaved_refresh = None

    async def _exchange_refresh_token(self, refresh: str) -> RefreshedTokens:
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
        return RefreshedTokens(
            used_refresh_token=refresh,
            access_token=tokens["access_token"],
            refresh_token=tokens.get("refresh_token", refresh),
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
    ) -> tuple[_CallbackServer, list[str], list[str], int]:
        auth_code: list[str] = []
        oauth_error: list[str] = []

        class CallbackHandler(BaseHTTPRequestHandler):
            timeout = CALLBACK_READ_TIMEOUT_SECONDS

            def setup(self) -> None:
                super().setup()
                self.rfile.close()
                self.rfile = io.BufferedReader(
                    _DeadlineReader(
                        self.connection,
                        per_read=CALLBACK_READ_TIMEOUT_SECONDS,
                        total=CALLBACK_REQUEST_SECONDS,
                    )
                )

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                state = params.get("state", [""])[0]
                if parsed.path != "/callback":
                    self.send_response(404)
                    message = b"Not found"
                elif not secrets.compare_digest(state.encode(), expected_state.encode()):
                    # Not this sign-in's redirect: a tab left from an earlier
                    # attempt, or any page the browser loaded requesting this
                    # URL. Refused, and the wait goes on, so neither can end
                    # the sign-in.
                    self.send_response(400)
                    message = (
                        b"This sign-in link is not from the current attempt. "
                        b"Finish signing in from the newest browser tab."
                    )
                elif "error" in params:
                    oauth_error.append(_oauth_error_text(params))
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
        # The OAuth app lists 41899-41910 as redirect URIs; a port past that is
        # rejected on the redirect. A configured port outside the range is the
        # only one known to be registered.
        in_range = CALLBACK_FIRST_PORT <= first_port <= CALLBACK_LAST_PORT
        last_port = CALLBACK_LAST_PORT if in_range else first_port
        for port in range(first_port, last_port + 1):
            try:
                server = _CallbackServer(("127.0.0.1", port), CallbackHandler)
            except OSError:
                continue
            try:
                server.companion = _CallbackServer6(("::1", port), CallbackHandler)
            except OSError as exc:
                if exc.errno not in _NO_IPV6_LOOPBACK:
                    # Another program listens on ::1 there (or on every
                    # address, dual-stack): a browser trying ::1 first
                    # would take it the code.
                    server.server_close()
                    continue
            return server, auth_code, oauth_error, port
        raise RuntimeError(f"Could not bind OAuth callback on ports {first_port}-{last_port}")

    @staticmethod
    def _wait_for_callback(
        server: _CallbackServer,
        auth_code: list[str],
        oauth_error: list[str],
        *,
        timeout: float = CALLBACK_TIMEOUT_SECONDS,
    ) -> str:
        """Serve requests until the redirect to /callback arrives or `timeout`
        passes. A request for another path, such as a browser fetching
        /favicon.ico, does not end the wait."""
        deadline = time.monotonic() + timeout
        servers = [server] if server.companion is None else [server, server.companion]
        try:
            with selectors.DefaultSelector() as selector:
                for each in servers:
                    selector.register(each, selectors.EVENT_READ)
                while not auth_code and not oauth_error:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    for key, _events in selector.select(remaining):
                        ready = key.fileobj
                        assert isinstance(ready, _CallbackServer)
                        # It has a connection waiting: take it without waiting.
                        ready.timeout = 0
                        ready.handle_request()
                        if auth_code or oauth_error:
                            break
        finally:
            server.server_close()

        if not auth_code:
            if oauth_error:
                raise OAuthDenied(oauth_error[0])
            raise ValueError("OAuth callback timed out")
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
