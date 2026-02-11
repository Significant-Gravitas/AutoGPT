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
    "More info: https://supabase.com/docs/guides/auth/signing-keys#choosing-the-right-signing-algorithm"  # noqa
)


class Settings:
    def __init__(self):
        self.JWT_VERIFY_KEY: str = os.getenv(
            "JWT_VERIFY_KEY", os.getenv("SUPABASE_JWT_SECRET", "")
        ).strip()
        self.JWT_ALGORITHM: str = os.getenv("JWT_SIGN_ALGORITHM", "HS256").strip()

        self.validate()

    def validate(self):
        if not self.JWT_VERIFY_KEY:
            raise AuthConfigError(
                "JWT_VERIFY_KEY must be set. "
                "An empty JWT secret would allow anyone to forge valid tokens."
            )

        if len(self.JWT_VERIFY_KEY) < 32:
            logger.warning(
                "⚠️ JWT_VERIFY_KEY appears weak (less than 32 characters). "
                "Consider using a longer, cryptographically secure secret."
            )

        supported_algorithms = get_default_algorithms().keys()

        if not has_crypto:
            logger.warning(
                "⚠️ Asymmetric JWT verification is not available "
                "because the 'cryptography' package is not installed. "
                + ALGO_RECOMMENDATION
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

        if self.JWT_ALGORITHM.startswith("HS"):
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
