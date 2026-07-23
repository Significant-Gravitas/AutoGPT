import logging
import os

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
                "JWT_JWKS_URL must be set. Better Auth issues asymmetric (ES256) "
                "tokens verified against the JWKS endpoint, so without it the "
                "backend cannot verify any live session. JWT_VERIFY_KEY only "
                "covers transient legacy (Supabase HS256) tokens during the "
                "cutover window and is optional."
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

        if self.JWT_VERIFY_KEY and len(self.JWT_VERIFY_KEY) < 32:
            logger.warning(
                "⚠️ JWT_VERIFY_KEY appears weak (less than 32 characters). "
                "Consider using a longer, cryptographically secure secret."
            )

        supported_algorithms = get_default_algorithms().keys()

        # JWT_JWKS_URL is required (checked above) and asymmetric (ES256)
        # verification needs the cryptography package, so a missing crypto lib
        # is always a hard error now — not the soft warning it was when a
        # JWKS-less HS256-only config was still allowed.
        if not has_crypto:
            raise AuthConfigError(
                "JWT_JWKS_URL is set but the 'cryptography' package is not "
                "installed, so asymmetric JWT verification is unavailable."
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
