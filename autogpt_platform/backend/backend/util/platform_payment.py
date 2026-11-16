from backend.util.settings import Settings

# The one reader of the setting, so a test patches one place for every caller.
settings = Settings()


def is_platform_payment_enabled() -> bool:
    """Whether an account without a plan is paywalled (``ENABLE_PLATFORM_PAYMENT``)."""
    return settings.config.enable_platform_payment
