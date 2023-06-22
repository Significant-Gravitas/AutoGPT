from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "0.0.0.0"
    port: int = 6060
    workers_count: int = 1
    reload: bool = True
    factory: bool = True


settings = Settings()
