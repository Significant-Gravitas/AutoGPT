from backend.util.settings import Settings

settings = Settings()


def get_frontend_base_url() -> str:
    return (
        settings.config.frontend_base_url or settings.config.platform_base_url
    ).rstrip("/")
