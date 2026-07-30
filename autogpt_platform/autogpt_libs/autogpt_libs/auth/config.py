import logging
import os
from urllib.parse import urlparse

from jwt.algorithms import get_default_algorithms, has_crypto

logger = logging.getLogger(__name__)


class AuthConfigError(ValueError):
    """Raised when authentication configuration is invalid."""

    pass


ALGO_RECOMMENDATION = (
    "We highly recommend using an asymmetric algorithm such as ES256, "
    "because when leaked, a shared secret would allow anyone to "
    "forge valid tokens and impersonate users. "
    "Configure JWT_JWKS_URL to verify asymmetric tokens issued by the "
    "platform auth service."
)


class Settings:
    def __init__(self):
        self.JWT_VERIFY_KEY: str = os.getenv(
            "JWT_VERIFY_KEY", os.getenv("SUPABASE_JWT_SECRET", "")
        ).strip()
        self.JWT_ALGORITHM: str = os.getenv("JWT_SIGN_ALGORITHM", "HS256").strip()
        self.JWT_JWKS_URL: str = os.getenv("JWT_JWKS_URL", "").strip()
        self.JWT_JWKS_ALGORITHMS: list[str] = [
            algo.strip()
            for algo in os.getenv("JWT_JWKS_ALGORITHMS", "ES256,RS256,EdDSA").split(",")
            if algo.strip()
        ]

        self.validate()

    def validate(self):
        if not self.JWT_JWKS_URL:
            raise AuthConfigError(
                "JWT_JWKS_URL must be set to the JWKS endpoint of the platform "
                "auth service (e.g. https://<frontend-host>/api/auth/jwks). "
                "Better Auth issues asymmetric (ES256) tokens verified against "
                "that endpoint, so without it the backend cannot verify any live "
                "session. JWT_VERIFY_KEY is not a substitute: it only covers "
                "transient legacy (Supabase HS256) tokens during the cutover "
                "window."
            )

        if self.JWT_JWKS_URL and not self.JWT_JWKS_URL.startswith(
            ("http://", "https://")
        ):
            # Caught here rather than as a cryptic PyJWKClientError on the
            # first request that hits the JWKS path.
            raise AuthConfigError(
                f"Invalid JWT_JWKS_URL: '{self.JWT_JWKS_URL}'. "
                "Must be an http:// or https:// URL."
            )

        if self.JWT_JWKS_URL.startswith("http://"):
            try:
                hostname = urlparse(self.JWT_JWKS_URL).hostname or ""
            except ValueError as e:
                # Malformed netloc, e.g. an unbalanced IPv6 bracket.
                raise AuthConfigError(
                    f"Invalid JWT_JWKS_URL: '{self.JWT_JWKS_URL}': {e}"
                ) from e
            # A single-label host is a container/service name on a private
            # network (e.g. http://frontend:3000), as trusted as loopback.
            host_is_local = hostname in ("localhost", "127.0.0.1", "::1") or (
                "." not in hostname and ":" not in hostname
            )
            allow_insecure = os.getenv(
                "JWKS_ALLOW_INSECURE_TRANSPORT", ""
            ).strip().lower() in ("1", "true", "yes")
            if not host_is_local and not allow_insecure:
                raise AuthConfigError(
                    "JWT_JWKS_URL fetches keys over cleartext http:// from "
                    f"a non-local host ('{hostname}'). The backend trusts "
                    "whatever keys that URL returns, so an attacker in the "
                    "path could substitute them and forge tokens for any "
                    "user. Use https://, or set "
                    "JWKS_ALLOW_INSECURE_TRANSPORT=true if the network path "
                    "is trusted."
                )
            if not host_is_local:
                logger.warning(
                    "⚠️ JWKS_ALLOW_INSECURE_TRANSPORT is enabled: JWKS keys "
                    f"are fetched over cleartext http:// from '{hostname}'. "
                    "Anyone in the network path can substitute them and forge "
                    "tokens. Use https:// unless the path is fully trusted."
                )

        if self.JWT_VERIFY_KEY and len(self.JWT_VERIFY_KEY) < 32:
            logger.warning(
                "⚠️ JWT_VERIFY_KEY appears weak (less than 32 characters). "
                "Consider using a longer, cryptographically secure secret."
            )

        supported_algorithms = get_default_algorithms().keys()

        if not has_crypto:
            raise AuthConfigError(
                "'cryptography' package is required for JWT verification "
                "but not installed"
            )

        if (
            self.JWT_ALGORITHM not in supported_algorithms
            or self.JWT_ALGORITHM == "none"
        ):
            raise AuthConfigError(
                f"Invalid JWT_SIGN_ALGORITHM: '{self.JWT_ALGORITHM}'. "
                "Supported algorithms are listed on "
                "https://pyjwt.readthedocs.io/en/stable/algorithms.html"
            )

        for algo in self.JWT_JWKS_ALGORITHMS:
            if (
                algo not in supported_algorithms
                or algo == "none"
                or algo.startswith("HS")
            ):
                raise AuthConfigError(
                    f"Invalid JWT_JWKS_ALGORITHMS entry: '{algo}'. "
                    "JWKS verification only supports asymmetric algorithms."
                )

        if "ES256" not in self.JWT_JWKS_ALGORITHMS:
            # Advisory, not fatal: a modified auth service could sign with
            # something else, but the stock platform frontend signs ES256 —
            # excluding it here silently rejects every token it issues.
            logger.warning(
                "⚠️ JWT_JWKS_ALGORITHMS does not include ES256, which is the "
                "algorithm the platform auth service signs with. With this "
                "setting, every token it issues will be rejected. Add ES256 "
                "unless a customized auth service signs with a different "
                "algorithm."
            )

        if self.JWT_VERIFY_KEY and self.JWT_ALGORITHM.startswith("HS"):
            logger.warning(
                f"⚠️ JWT_SIGN_ALGORITHM is set to '{self.JWT_ALGORITHM}', "
                "a symmetric shared-key signature algorithm. " + ALGO_RECOMMENDATION
            )


_settings: Settings = None  # type: ignore


def get_settings() -> Settings:
    global _settings

    if not _settings:
        _settings = Settings()

    return _settings


def verify_settings() -> None:
    global _settings

    if not _settings:
        _settings = Settings()  # calls validation indirectly
        return

    _settings.validate()
