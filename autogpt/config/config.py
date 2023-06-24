"""Configuration class to store the state of bools for different scripts access."""
import os
import re
from typing import List

import openai
import yaml
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from colorama import Fore

import autogpt


class Config:
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self) -> None:
        """Initialize the Config class"""
        self.workspace_path: str = None
        self.file_logger_path: str = None

        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0
        self.speak_mode = False
        self.skip_reprompt = False
        self.allow_downloads = False
        self.skip_news = False

        self.authorise_key = os.getenv("AUTHORISE_COMMAND_KEY", "y")
        self.exit_key = os.getenv("EXIT_KEY", "n")
        self.plain_output = os.getenv("PLAIN_OUTPUT", "False") == "True"

        disabled_command_categories = os.getenv("DISABLED_COMMAND_CATEGORIES")
        if disabled_command_categories:
            self.disabled_command_categories = disabled_command_categories.split(",")
        else:
            self.disabled_command_categories = []

        self.shell_command_control = os.getenv("SHELL_COMMAND_CONTROL", "denylist")

        # DENY_COMMANDS is deprecated and included for backwards-compatibility
        shell_denylist = os.getenv("SHELL_DENYLIST", os.getenv("DENY_COMMANDS"))
        if shell_denylist:
            self.shell_denylist = shell_denylist.split(",")
        else:
            self.shell_denylist = ["sudo", "su"]

        # ALLOW_COMMANDS is deprecated and included for backwards-compatibility
        shell_allowlist = os.getenv("SHELL_ALLOWLIST", os.getenv("ALLOW_COMMANDS"))
        if shell_allowlist:
            self.shell_allowlist = shell_allowlist.split(",")
        else:
            self.shell_allowlist = []

        self.ai_settings_file = os.getenv("AI_SETTINGS_FILE", "ai_settings.yaml")
        self.prompt_settings_file = os.getenv(
            "PROMPT_SETTINGS_FILE", "prompt_settings.yaml"
        )
        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo")
        self.smart_llm_model = os.getenv("SMART_LLM_MODEL", "gpt-3.5-turbo")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

        self.browse_spacy_language_model = os.getenv(
            "BROWSE_SPACY_LANGUAGE_MODEL", "en_core_web_sm"
        )

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_organization = os.getenv("OPENAI_ORGANIZATION")
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.use_azure = os.getenv("USE_AZURE") == "True"
        self.execute_local_commands = (
            os.getenv("EXECUTE_LOCAL_COMMANDS", "False") == "True"
        )
        self.restrict_to_workspace = (
            os.getenv("RESTRICT_TO_WORKSPACE", "True") == "True"
        )

        if self.use_azure:
            self.load_azure_config()
            openai.api_type = self.openai_api_type
            openai.api_base = self.openai_api_base
            openai.api_version = self.openai_api_version
        elif os.getenv("OPENAI_API_BASE_URL", None):
            openai.api_base = os.getenv("OPENAI_API_BASE_URL")

        if self.openai_organization is not None:
            openai.organization = self.openai_organization

        self.openai_functions = os.getenv("OPENAI_FUNCTIONS", "False") == "True"

        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        # ELEVENLABS_VOICE_1_ID is deprecated and included for backwards-compatibility
        self.elevenlabs_voice_id = os.getenv(
            "ELEVENLABS_VOICE_ID", os.getenv("ELEVENLABS_VOICE_1_ID")
        )
        self.streamelements_voice = os.getenv("STREAMELEMENTS_VOICE", "Brian")

        # Backwards-compatibility shim for deprecated env variables
        if os.getenv("USE_MAC_OS_TTS"):
            default_tts_provider = "macos"
        elif self.elevenlabs_api_key:
            default_tts_provider = "elevenlabs"
        elif os.getenv("USE_BRIAN_TTS"):
            default_tts_provider = "streamelements"
        else:
            default_tts_provider = "gtts"

        self.text_to_speech_provider = os.getenv(
            "TEXT_TO_SPEECH_PROVIDER", default_tts_provider
        )

        self.github_api_key = os.getenv("GITHUB_API_KEY")
        self.github_username = os.getenv("GITHUB_USERNAME")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        # CUSTOM_SEARCH_ENGINE_ID is deprecated and included for backwards-compatibility
        self.google_custom_search_engine_id = os.getenv(
            "GOOGLE_CUSTOM_SEARCH_ENGINE_ID", os.getenv("CUSTOM_SEARCH_ENGINE_ID")
        )

        self.image_provider = os.getenv("IMAGE_PROVIDER")
        self.image_size = int(os.getenv("IMAGE_SIZE", 256))
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.huggingface_image_model = os.getenv(
            "HUGGINGFACE_IMAGE_MODEL", "CompVis/stable-diffusion-v1-4"
        )
        self.audio_to_text_provider = os.getenv("AUDIO_TO_TEXT_PROVIDER", "huggingface")
        self.huggingface_audio_to_text_model = os.getenv(
            "HUGGINGFACE_AUDIO_TO_TEXT_MODEL"
        )
        self.sd_webui_url = os.getenv("SD_WEBUI_URL", "http://localhost:7860")
        self.sd_webui_auth = os.getenv("SD_WEBUI_AUTH")

        # Selenium browser settings
        self.selenium_web_browser = os.getenv("USE_WEB_BROWSER", "chrome")
        self.selenium_headless = os.getenv("HEADLESS_BROWSER", "True") == "True"

        # User agent header to use when making HTTP requests
        # Some websites might just completely deny request with an error code if
        # no user agent was found.
        self.user_agent = os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
        )

        self.memory_backend = os.getenv("MEMORY_BACKEND", "json_file")
        self.memory_index = os.getenv("MEMORY_INDEX", "auto-gpt-memory")

        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.wipe_redis_on_start = os.getenv("WIPE_REDIS_ON_START", "True") == "True"

        self.plugins_dir = os.getenv("PLUGINS_DIR", "plugins")
        self.plugins: List[AutoGPTPluginTemplate] = []
        self.plugins_openai = []

        # Deprecated. Kept for backwards-compatibility. Will remove in a future version.
        plugins_allowlist = os.getenv("ALLOWLISTED_PLUGINS")
        if plugins_allowlist:
            self.plugins_allowlist = plugins_allowlist.split(",")
        else:
            self.plugins_allowlist = []

        # Deprecated. Kept for backwards-compatibility. Will remove in a future version.
        plugins_denylist = os.getenv("DENYLISTED_PLUGINS")
        if plugins_denylist:
            self.plugins_denylist = plugins_denylist.split(",")
        else:
            self.plugins_denylist = []

        # Avoid circular imports
        from autogpt.plugins import DEFAULT_PLUGINS_CONFIG_FILE

        self.plugins_config_file = os.getenv(
            "PLUGINS_CONFIG_FILE", DEFAULT_PLUGINS_CONFIG_FILE
        )
        self.load_plugins_config()

        self.chat_messages_enabled = os.getenv("CHAT_MESSAGES_ENABLED") == "True"

    def load_plugins_config(self) -> "autogpt.plugins.PluginsConfig":
        # Avoid circular import
        from autogpt.plugins.plugins_config import PluginsConfig

        self.plugins_config = PluginsConfig.load_config(global_config=self)
        return self.plugins_config

    def get_azure_deployment_id_for_model(self, model: str) -> str:
        """
        Returns the relevant deployment id for the model specified.

        Parameters:
            model(str): The model to map to the deployment id.

        Returns:
            The matching deployment id if found, otherwise an empty string.
        """
        if model == self.fast_llm_model:
            return self.azure_model_to_deployment_id_map[
                "fast_llm_model_deployment_id"
            ]  # type: ignore
        elif model == self.smart_llm_model:
            return self.azure_model_to_deployment_id_map[
                "smart_llm_model_deployment_id"
            ]  # type: ignore
        elif model == "text-embedding-ada-002":
            return self.azure_model_to_deployment_id_map[
                "embedding_model_deployment_id"
            ]  # type: ignore
        else:
            return ""

    AZURE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "../..", "azure.yaml")

    def load_azure_config(self, config_file: str = AZURE_CONFIG_FILE) -> None:
        """
        Loads the configuration parameters for Azure hosting from the specified file
          path as a yaml file.

        Parameters:
            config_file(str): The path to the config yaml file. DEFAULT: "../azure.yaml"

        Returns:
            None
        """
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader) or {}
        self.openai_api_type = config_params.get("azure_api_type") or "azure"
        self.openai_api_base = config_params.get("azure_api_base") or ""
        self.openai_api_version = (
            config_params.get("azure_api_version") or "2023-03-15-preview"
        )
        self.azure_model_to_deployment_id_map = config_params.get("azure_model_map", {})

    def set_continuous_mode(self, value: bool) -> None:
        """Set the continuous mode value."""
        self.continuous_mode = value

    def set_continuous_limit(self, value: int) -> None:
        """Set the continuous limit value."""
        self.continuous_limit = value

    def set_speak_mode(self, value: bool) -> None:
        """Set the speak mode value."""
        self.speak_mode = value

    def set_fast_llm_model(self, value: str) -> None:
        """Set the fast LLM model value."""
        self.fast_llm_model = value

    def set_smart_llm_model(self, value: str) -> None:
        """Set the smart LLM model value."""
        self.smart_llm_model = value

    def set_embedding_model(self, value: str) -> None:
        """Set the model to use for creating embeddings."""
        self.embedding_model = value

    def set_openai_api_key(self, value: str) -> None:
        """Set the OpenAI API key value."""
        self.openai_api_key = value

    def set_elevenlabs_api_key(self, value: str) -> None:
        """Set the ElevenLabs API key value."""
        self.elevenlabs_api_key = value

    def set_elevenlabs_voice_1_id(self, value: str) -> None:
        """Set the ElevenLabs Voice 1 ID value."""
        self.elevenlabs_voice_id = value

    def set_elevenlabs_voice_2_id(self, value: str) -> None:
        """Set the ElevenLabs Voice 2 ID value."""
        self.elevenlabs_voice_2_id = value

    def set_google_api_key(self, value: str) -> None:
        """Set the Google API key value."""
        self.google_api_key = value

    def set_custom_search_engine_id(self, value: str) -> None:
        """Set the custom search engine id value."""
        self.google_custom_search_engine_id = value

    def set_debug_mode(self, value: bool) -> None:
        """Set the debug mode value."""
        self.debug_mode = value

    def set_plugins(self, value: list) -> None:
        """Set the plugins value."""
        self.plugins = value

    def set_temperature(self, value: int) -> None:
        """Set the temperature value."""
        self.temperature = value

    def set_memory_backend(self, name: str) -> None:
        """Set the memory backend name."""
        self.memory_backend = name


def check_openai_api_key(config: Config) -> None:
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    if not config.openai_api_key:
        print(
            Fore.RED
            + "Please set your OpenAI API key in .env or as an environment variable."
            + Fore.RESET
        )
        print("You can get your key from https://platform.openai.com/account/api-keys")
        openai_api_key = input(
            "If you do have the key, please enter your OpenAI API key now:\n"
        )
        key_pattern = r"^sk-\w{48}"
        openai_api_key = openai_api_key.strip()
        if re.search(key_pattern, openai_api_key):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            cfg.set_openai_api_key(openai_api_key)
            print(
                Fore.GREEN
                + "OpenAI API key successfully set!\n"
                + Fore.ORANGE
                + "NOTE: The API key you've set is only temporary.\n"
                + "For longer sessions, please set it in .env file"
                + Fore.RESET
            )
        else:
            print("Invalid OpenAI API key!")
            exit(1)
