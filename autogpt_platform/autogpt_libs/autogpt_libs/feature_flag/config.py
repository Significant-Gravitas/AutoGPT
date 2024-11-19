import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    SDK_KEY: str = os.getenv("LAUNCH_DARKLY_SDK_KEY", "")


settings = Settings()