import os

from forge.file_storage import FileStorageBackendName
from forge.models.config import SystemSettings, UserConfigurable
from forge.speech.say import TTSConfig


class BaseConfig(SystemSettings):
    name: str = "Base configuration"
    description: str = "Default configuration for forge agent."

    # TTS configuration
    tts_config: TTSConfig = TTSConfig()

    # File storage
    file_storage_backend: FileStorageBackendName = UserConfigurable(
        default=FileStorageBackendName.LOCAL, from_env="FILE_STORAGE_BACKEND"
    )

    # Shell commands
    execute_local_commands: bool = UserConfigurable(
        default=False,
        from_env=lambda: os.getenv("EXECUTE_LOCAL_COMMANDS", "False") == "True",
    )
