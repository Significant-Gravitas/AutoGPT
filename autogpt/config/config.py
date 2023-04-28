"""Configuration class to store the state of bools for different scripts access."""
import os
from typing import List

import openai
import yaml
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from colorama import Fore

from autogpt.singleton import Singleton


class Config(metaclass=Singleton):
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self) -> None:
        """Initialize the Config class"""
        self.workspace_path = None
        self.file_logger_path = None

        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0
        self.speak_mode = False
        self.skip_reprompt = False
        self.allow_downloads = False
        self.skip_news = False

        self.authorise_key = os.getenv("AUTHORISE_COMMAND_KEY", "y")
        self.exit_key = os.getenv("EXIT_KEY", "n")
        self.ai_settings_file = os.getenv("AI_SETTINGS_FILE", "ai_settings.yaml")
        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo")
        self.smart_llm_model = os.getenv("SMART_LLM_MODEL", "gpt-4")
        self.fast_token_limit = int(os.getenv("FAST_TOKEN_LIMIT", 4000))
        self.smart_token_limit = int(os.getenv("SMART_TOKEN_LIMIT", 8000))
        self.browse_chunk_max_length = int(os.getenv("BROWSE_CHUNK_MAX_LENGTH", 3000))
        self.browse_spacy_language_model = os.getenv(
            "BROWSE_SPACY_LANGUAGE_MODEL", "en_core_web_sm"
        )

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
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

        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_1_id = os.getenv("ELEVENLABS_VOICE_1_ID")
        self.elevenlabs_voice_2_id = os.getenv("ELEVENLABS_VOICE_2_ID")

        self.use_mac_os_tts = False
        self.use_mac_os_tts = os.getenv("USE_MAC_OS_TTS")

        self.chat_messages_enabled = os.getenv("CHAT_MESSAGES_ENABLED") == "True"

        self.use_brian_tts = False
        self.use_brian_tts = os.getenv("USE_BRIAN_TTS")

        self.github_api_key = os.getenv("GITHUB_API_KEY")
        self.github_username = os.getenv("GITHUB_USERNAME")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.custom_search_engine_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_region = os.getenv("PINECONE_ENV")

        self.weaviate_host = os.getenv("WEAVIATE_HOST")
        self.weaviate_port = os.getenv("WEAVIATE_PORT")
        self.weaviate_protocol = os.getenv("WEAVIATE_PROTOCOL", "http")
        self.weaviate_username = os.getenv("WEAVIATE_USERNAME", None)
        self.weaviate_password = os.getenv("WEAVIATE_PASSWORD", None)
        self.weaviate_scopes = os.getenv("WEAVIATE_SCOPES", None)
        self.weaviate_embedded_path = os.getenv("WEAVIATE_EMBEDDED_PATH")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY", None)
        self.use_weaviate_embedded = (
            os.getenv("USE_WEAVIATE_EMBEDDED", "False") == "True"
        )

        # milvus or zilliz cloud configuration.
        self.milvus_addr = os.getenv("MILVUS_ADDR", "localhost:19530")
        self.milvus_username = os.getenv("MILVUS_USERNAME")
        self.milvus_password = os.getenv("MILVUS_PASSWORD")
        self.milvus_collection = os.getenv("MILVUS_COLLECTION", "autogpt")
        self.milvus_secure = os.getenv("MILVUS_SECURE") == "True"

        self.image_provider = os.getenv("IMAGE_PROVIDER")
        self.image_size = int(os.getenv("IMAGE_SIZE", 256))
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.huggingface_image_model = os.getenv(
            "HUGGINGFACE_IMAGE_MODEL", "CompVis/stable-diffusion-v1-4"
        )
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

        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = os.getenv("REDIS_PORT", "6379")
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.wipe_redis_on_start = os.getenv("WIPE_REDIS_ON_START", "True") == "True"
        self.memory_index = os.getenv("MEMORY_INDEX", "auto-gpt")
        # Note that indexes must be created on db 0 in redis, this is not configurable.

        self.memory_backend = os.getenv("MEMORY_BACKEND", "local")

        self.plugins_dir = os.getenv("PLUGINS_DIR", "plugins")
        self.plugins: List[AutoGPTPluginTemplate] = []
        self.plugins_openai = []

        plugins_allowlist = os.getenv("ALLOWLISTED_PLUGINS")
        if plugins_allowlist:
            self.plugins_allowlist = plugins_allowlist.split(",")
        else:
            self.plugins_allowlist = []
        self.plugins_denylist = []

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
            config_params = yaml.load(file, Loader=yaml.FullLoader)
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

    def set_fast_token_limit(self, value: int) -> None:
        """Set the fast token limit value."""
        self.fast_token_limit = value

    def set_smart_token_limit(self, value: int) -> None:
        """Set the smart token limit value."""
        self.smart_token_limit = value

    def set_browse_chunk_max_length(self, value: int) -> None:
        """Set the browse_website command chunk max length value."""
        self.browse_chunk_max_length = value

    def set_openai_api_key(self, value: str) -> None:
        """Set the OpenAI API key value."""
        self.openai_api_key = value

    def set_elevenlabs_api_key(self, value: str) -> None:
        """Set the ElevenLabs API key value."""
        self.elevenlabs_api_key = value

    def set_elevenlabs_voice_1_id(self, value: str) -> None:
        """Set the ElevenLabs Voice 1 ID value."""
        self.elevenlabs_voice_1_id = value

    def set_elevenlabs_voice_2_id(self, value: str) -> None:
        """Set the ElevenLabs Voice 2 ID value."""
        self.elevenlabs_voice_2_id = value

    def set_google_api_key(self, value: str) -> None:
        """Set the Google API key value."""
        self.google_api_key = value

    def set_custom_search_engine_id(self, value: str) -> None:
        """Set the custom search engine id value."""
        self.custom_search_engine_id = value

    def set_pinecone_api_key(self, value: str) -> None:
        """Set the Pinecone API key value."""
        self.pinecone_api_key = value

    def set_pinecone_region(self, value: str) -> None:
        """Set the Pinecone region value."""
        self.pinecone_region = value

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


def check_openai_api_key() -> None:
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    cfg = Config()
    if not cfg.openai_api_key:
        print(
            Fore.RED
            + "Please set your OpenAI API key in .env or as an environment variable."
            + Fore.RESET
        )
        print("You can get your key from https://platform.openai.com/account/api-keys")
        exit(1)
