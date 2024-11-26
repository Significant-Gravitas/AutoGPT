import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    LAUNCH_DARKLY_SDK_KEY: str = os.getenv("LAUNCH_DARKLY_SDK_KEY", "")


SETTINGS = Settings()
