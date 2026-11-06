"""
Consent tokens — shim-enforced, not platform-asserted (§9).

`REQUEST_RECORDING_CONSENT` asks the shim to show a local confirmation. Only an
affirmative response mints the token required by `START_RECORDING`; a platform
that sends START without a valid token gets CONSENT_REQUIRED.

Design (per docs/WORKFLOW_RECORDING.md §9):

- The token is minted **only** by a shim-local call (`issue_consent_token`),
  simulating the OS-native confirmation. The platform can never mint one — it
  has no access to the shim-local HMAC key and never sees the issuance path.
- `validate_consent_token` checks the signature, expiry, and single-use status.
  A token is consumed on the first successful START so it can't be replayed.
- `request_user_consent()` renders the confirmation in the shim process, not
  the platform-controlled browser, and only mints after an affirmative click.

The token is intentionally opaque to the platform: it round-trips it verbatim.
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import secrets
import time
from hashlib import sha256

logger = logging.getLogger(__name__)

# How long an issued token stays valid before it must be re-requested. Short by
# design: consent is per-recording and the user just clicked "record".
CONSENT_TOKEN_TTL_SECONDS: float = 300.0

# Token format version, so we can evolve the encoding without ambiguity.
_TOKEN_VERSION = 1


class ConsentError(Exception):
    """Raised when a consent token is missing, malformed, expired, or reused.

    Callers translate this into the CONSENT_REQUIRED wire error (§9).
    """


class ConsentBroker:
    """Issues and validates single-use, shim-local consent tokens.

    Holds a process-lifetime HMAC key (generated on construction) and the set
    of already-consumed token ids. Both are in-memory only: a shim restart
    invalidates outstanding tokens, which is the conservative choice for a
    consent gate — better to re-ask than to honor a token minted by a previous
    process the user may not remember authorizing.
    """

    def __init__(self) -> None:
        # Per-process secret. The platform never sees this, so it cannot forge
        # a token. Rotating it on restart is deliberate (see class docstring).
        self._key = secrets.token_bytes(32)
        self._consumed: set[str] = set()

    # ── Issuance (shim-local only) ────────────────────────────────────────

    def issue_consent_token(
        self,
        *,
        mode: str,
        interpretation_route: str,
        ttl_seconds: float = CONSENT_TOKEN_TTL_SECONDS,
    ) -> str:
        """Mint a signed, single-use consent token.

        This is the ONLY way a valid token comes into existence, and it is a
        shim-local call — never reachable from the wire. `mode` and
        `interpretation_route` are bound into the token so a token issued for
        one kind of recording can't be replayed for another.
        """
        body = {
            "v": _TOKEN_VERSION,
            "jti": secrets.token_hex(16),  # unique id, drives single-use
            "mode": mode,
            "route": interpretation_route,
            "iat": time.time(),
            "exp": time.time() + ttl_seconds,
        }
        payload = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
        sig = hmac.new(self._key, payload, sha256).digest()  # always 32 bytes
        # Append the fixed-length 32-byte HMAC and base64 the whole thing. We
        # do NOT use a delimiter: the digest is random bytes and could contain
        # any separator we picked, so we split by the known sig length instead.
        token = base64.urlsafe_b64encode(payload + sig).decode("ascii")
        logger.debug("Issued consent token jti=%s mode=%s", body["jti"], mode)
        return token

    # ── Validation (consumes the token on success) ────────────────────────

    def validate_consent_token(
        self,
        token: str | None,
        *,
        mode: str | None = None,
        interpretation_route: str | None = None,
    ) -> dict:
        """Validate + consume a token. Returns the decoded body on success.

        Raises ConsentError when the token is missing, malformed, has a bad
        signature, is expired, has already been consumed, or doesn't match the
        requested `mode` / `interpretation_route` (when those are supplied).

        On success the token's jti is recorded so a second START with the same
        token fails — single-use is what stops the platform from replaying one
        affirmative consent across many recordings.
        """
        if not token:
            raise ConsentError("no consent token supplied")
        try:
            blob = base64.urlsafe_b64decode(token.encode("ascii"))
        except (ValueError, TypeError) as exc:
            raise ConsentError(f"malformed consent token: {exc}") from exc
        # The trailing 32 bytes are the HMAC-SHA256 digest; everything before is
        # the JSON payload. A blob shorter than the digest can't be ours.
        if len(blob) <= sha256().digest_size:
            raise ConsentError("malformed consent token: too short")
        payload, sig = blob[: -sha256().digest_size], blob[-sha256().digest_size :]

        expected = hmac.new(self._key, payload, sha256).digest()
        if not hmac.compare_digest(expected, sig):
            # Wrong signature ⇒ not minted by this shim process. This is the
            # case where the platform tried to self-assert consent.
            raise ConsentError("consent token signature invalid (not shim-issued)")

        try:
            body = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ConsentError(f"unreadable consent token body: {exc}") from exc

        if body.get("v") != _TOKEN_VERSION:
            raise ConsentError(f"unsupported consent token version: {body.get('v')!r}")
        if time.time() > float(body.get("exp", 0)):
            raise ConsentError("consent token expired")

        jti = body.get("jti")
        if not isinstance(jti, str):
            raise ConsentError("consent token missing jti")
        if jti in self._consumed:
            raise ConsentError("consent token already used (single-use)")

        if mode is not None and body.get("mode") != mode:
            raise ConsentError("consent token mode does not match request")
        if interpretation_route is not None and body.get("route") != interpretation_route:
            raise ConsentError("consent token route does not match request")

        self._consumed.add(jti)
        return body


# ── OS-native dialog seam ─────────────────────────────────────────────────────


def request_user_consent(
    broker: ConsentBroker,
    *,
    mode: str,
    interpretation_route: str,
) -> str | None:
    """Show the OS-native confirmation and return a fresh consent token.

    Returns ``None`` when the prompt is declined or cannot be rendered. A
    headless shim therefore fails closed instead of silently authorizing a
    recording.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if interpretation_route == "screenshots_to_cloud":
            detail = (
                "After you stop, the structured trajectory and screenshots may be sent "
                "to AutoGPT for browser review and skill generation."
            )
        else:
            detail = (
                "Screen pixels stay on this machine. After you stop, a hygiene-redacted "
                "structured trajectory is sent through AutoGPT to your authenticated "
                "browser for review."
            )
        approved = messagebox.askyesno(
            "Allow AutoGPT workflow recording?",
            f"AutoGPT wants to start a {mode} recording.\n\n{detail}",
            parent=root,
        )
        root.destroy()
    except Exception:
        logger.warning("Could not render the local recording-consent prompt", exc_info=True)
        return None
    if not approved:
        return None
    return broker.issue_consent_token(mode=mode, interpretation_route=interpretation_route)


__all__ = [
    "CONSENT_TOKEN_TTL_SECONDS",
    "ConsentBroker",
    "ConsentError",
    "request_user_consent",
]
