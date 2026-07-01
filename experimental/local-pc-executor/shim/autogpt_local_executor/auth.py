"""
OAuth authentication for the shim.

Uses AutoGPT's existing OAuth 2.0 provider with:
- Authorization Code + PKCE flow
- Localhost callback server on port 41899
- Token storage in OS keychain via the `keyring` library

See docs/OAUTH_FLOW.md for the full sequence diagram.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

# TODO: pip install keyring httpx
# import keyring
# import httpx


KEYCHAIN_SERVICE = "autogpt-local-executor"
KEYCHAIN_ACCESS_TOKEN_KEY = "access_token"
KEYCHAIN_REFRESH_TOKEN_KEY = "refresh_token"


class KeychainTokenStore:
    """
    Stores OAuth tokens in the OS keychain using the `keyring` library.

    Backends by platform:
        macOS   → Keychain Services
        Linux   → Secret Service (GNOME Keyring / KWallet)
        Windows → Windows Credential Manager
    """

    def get_access_token(self) -> Optional[str]:
        """Return the stored access token, or None if not authenticated."""
        # TODO: return keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCESS_TOKEN_KEY)
        raise NotImplementedError("Keychain token store not yet implemented")

    def store_tokens(self, access_token: str, refresh_token: str) -> None:
        """Persist both tokens to the OS keychain."""
        # TODO:
        # keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCESS_TOKEN_KEY, access_token)
        # keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_REFRESH_TOKEN_KEY, refresh_token)
        raise NotImplementedError

    def clear_tokens(self) -> None:
        """Remove all stored tokens (used by `autogpt-shim revoke`)."""
        # TODO:
        # keyring.delete_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCESS_TOKEN_KEY)
        # keyring.delete_password(KEYCHAIN_SERVICE, KEYCHAIN_REFRESH_TOKEN_KEY)
        raise NotImplementedError


class OAuthFlow:
    """
    Runs the Authorization Code + PKCE flow against the AutoGPT OAuth provider.

    Flow:
        1. generate_pkce_pair()   → code_verifier, code_challenge
        2. build_auth_url()       → URL to open in browser
        3. wait_for_callback()    → spins up localhost:PORT, waits for redirect
        4. exchange_code()        → POST /auth/token → access_token, refresh_token
        5. token_store.store()    → save to OS keychain
    """

    SCOPES = [
        "local_executor:connect",
        "local_executor:shell",
        "local_executor:files",
    ]

    def __init__(self, config: Any, token_store: KeychainTokenStore) -> None:
        self.config = config
        self.token_store = token_store

    def run(self) -> None:
        """
        Execute the full auth flow interactively.
        Opens the browser, waits for the user to approve, stores tokens.
        """
        code_verifier, code_challenge = self._generate_pkce_pair()
        auth_url = self._build_auth_url(code_challenge)

        print(f"\nOpening browser for AutoGPT authentication...")
        print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")
        webbrowser.open(auth_url)

        auth_code = self._wait_for_callback()
        tokens = self._exchange_code(auth_code, code_verifier)
        self.token_store.store_tokens(tokens["access_token"], tokens["refresh_token"])
        print("Authentication successful. Tokens stored in OS keychain.")

    @staticmethod
    def _generate_pkce_pair() -> tuple[str, str]:
        """Generate PKCE code_verifier and S256 code_challenge."""
        code_verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return code_verifier, code_challenge

    def _build_auth_url(self, code_challenge: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self.config.oauth_client_id,
            "redirect_uri": f"http://localhost:{self.config.oauth_redirect_port}/callback",
            "scope": " ".join(self.SCOPES),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": secrets.token_urlsafe(16),
        }
        return f"{self.config.platform_oauth_url}/authorize?" + urllib.parse.urlencode(params)

    def _wait_for_callback(self) -> str:
        """
        Spin up a temporary HTTP server on localhost to receive the OAuth callback.
        Returns the authorization code.
        """
        auth_code: list[str] = []

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                if "code" in params:
                    auth_code.append(params["code"][0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"<h1>Authenticated! You can close this tab.</h1>")

            def log_message(self, *args):
                pass  # suppress server logs

        server = HTTPServer(("localhost", self.config.oauth_redirect_port), CallbackHandler)
        server.handle_request()  # blocks until one request received
        server.server_close()

        if not auth_code:
            raise ValueError("No authorization code received from OAuth callback")
        return auth_code[0]

    def _exchange_code(self, auth_code: str, code_verifier: str) -> dict:
        """POST /auth/token to exchange the authorization code for tokens."""
        # TODO: implement with httpx
        # async with httpx.AsyncClient() as client:
        #     resp = await client.post(
        #         f"{self.config.platform_oauth_url}/token",
        #         data={
        #             "grant_type": "authorization_code",
        #             "code": auth_code,
        #             "redirect_uri": f"http://localhost:{self.config.oauth_redirect_port}/callback",
        #             "client_id": self.config.oauth_client_id,
        #             "code_verifier": code_verifier,
        #         },
        #     )
        #     resp.raise_for_status()
        #     return resp.json()
        raise NotImplementedError("Token exchange not yet implemented")
