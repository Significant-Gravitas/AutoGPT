import abc
import os

import openai
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractSingleton(abc.ABC, metaclass=Singleton):
    pass


class Config(metaclass=Singleton):
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self):
        """Initialize the Config class"""
        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0
        self.speak_mode = False
        self.skip_reprompt = False

        self.ai_settings_file = os.getenv("AI_SETTINGS_FILE", "ai_settings.yaml")
        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo")
        self.smart_llm_model = os.getenv("SMART_LLM_MODEL", "gpt-4")
        self.fast_token_limit = int(os.getenv("FAST_TOKEN_LIMIT", 4000))
        self.smart_token_limit = int(os.getenv("SMART_TOKEN_LIMIT", 8000))
        self.browse_chunk_max_length = int(os.getenv("BROWSE_CHUNK_MAX_LENGTH", 8192))
        self.browse_summary_max_token = int(os.getenv("BROWSE_SUMMARY_MAX_TOKEN", 300))

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.temperature = float(os.getenv("TEMPERATURE", "1"))
        self.use_azure = os.getenv("USE_AZURE") == "True"
        self.execute_local_commands = (
            os.getenv("EXECUTE_LOCAL_COMMANDS", "False") == "True"
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

        self.use_brian_tts = False
        self.use_brian_tts = os.getenv("USE_BRIAN_TTS")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.custom_search_engine_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_region = os.getenv("PINECONE_ENV")

        self.image_provider = os.getenv("IMAGE_PROVIDER")
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

        # User agent headers to use when browsing web
        # Some websites might just completely deny request with an error code if no user agent was found.
        self.user_agent = os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
        )
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = os.getenv("REDIS_PORT", "6379")
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.wipe_redis_on_start = os.getenv("WIPE_REDIS_ON_START", "True") == "True"
        self.memory_index = os.getenv("MEMORY_INDEX", "auto-gpt")
        # Note that indexes must be created on db 0 in redis, this is not configurable.

        self.memory_backend = os.getenv("MEMORY_BACKEND", "local")
        # Initialize the OpenAI API client
        openai.api_key = self.openai_api_key

    def get_azure_deployment_id_for_model(self, model: str) -> str:
        """
        Returns the relevant deployment id for the model specified.

        Parameters:
            model(str): The model to map to the deployment id.

        Returns:
            The matching deployment id if found, otherwise an empty string.
        """
        if model == self.fast_llm_model:
            return self.azure_model_to_deployment_id_map["fast_llm_model_deployment_id"]
        elif model == self.smart_llm_model:
            return self.azure_model_to_deployment_id_map[
                "smart_llm_model_deployment_id"
            ]
        elif model == "text-embedding-ada-002":
            return self.azure_model_to_deployment_id_map[
                "embedding_model_deployment_id"
            ]
        else:
            return ""

    AZURE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "azure.yaml")

    def load_azure_config(self, config_file: str = AZURE_CONFIG_FILE) -> None:
        """
        Loads the configuration parameters for Azure hosting from the specified file path as a yaml file.

        Parameters:
            config_file(str): The path to the config yaml file. DEFAULT: "../azure.yaml"

        Returns:
            None
        """
        try:
            with open(config_file) as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}
        self.openai_api_type = os.getenv(
            "OPENAI_API_TYPE", config_params.get("azure_api_type", "azure")
        )
        self.openai_api_base = os.getenv(
            "OPENAI_AZURE_API_BASE", config_params.get("azure_api_base", "")
        )
        self.openai_api_version = os.getenv(
            "OPENAI_AZURE_API_VERSION", config_params.get("azure_api_version", "")
        )
        self.azure_model_to_deployment_id_map = config_params.get("azure_model_map", [])

    def set_continuous_mode(self, value: bool):
        """Set the continuous mode value."""
        self.continuous_mode = value

    def set_continuous_limit(self, value: int):
        """Set the continuous limit value."""
        self.continuous_limit = value

    def set_speak_mode(self, value: bool):
        """Set the speak mode value."""
        self.speak_mode = value

    def set_fast_llm_model(self, value: str):
        """Set the fast LLM model value."""
        self.fast_llm_model = value

    def set_smart_llm_model(self, value: str):
        """Set the smart LLM model value."""
        self.smart_llm_model = value

    def set_fast_token_limit(self, value: int):
        """Set the fast token limit value."""
        self.fast_token_limit = value

    def set_smart_token_limit(self, value: int):
        """Set the smart token limit value."""
        self.smart_token_limit = value

    def set_browse_chunk_max_length(self, value: int):
        """Set the browse_website command chunk max length value."""
        self.browse_chunk_max_length = value

    def set_browse_summary_max_token(self, value: int):
        """Set the browse_website command summary max token value."""
        self.browse_summary_max_token = value

    def set_openai_api_key(self, value: str):
        """Set the OpenAI API key value."""
        self.openai_api_key = value

    def set_elevenlabs_api_key(self, value: str):
        """Set the ElevenLabs API key value."""
        self.elevenlabs_api_key = value

    def set_elevenlabs_voice_1_id(self, value: str):
        """Set the ElevenLabs Voice 1 ID value."""
        self.elevenlabs_voice_1_id = value

    def set_elevenlabs_voice_2_id(self, value: str):
        """Set the ElevenLabs Voice 2 ID value."""
        self.elevenlabs_voice_2_id = value

    def set_google_api_key(self, value: str):
        """Set the Google API key value."""
        self.google_api_key = value

    def set_custom_search_engine_id(self, value: str):
        """Set the custom search engine id value."""
        self.custom_search_engine_id = value

    def set_pinecone_api_key(self, value: str):
        """Set the Pinecone API key value."""
        self.pinecone_api_key = value

    def set_pinecone_region(self, value: str):
        """Set the Pinecone region value."""
        self.pinecone_region = value

    def set_debug_mode(self, value: bool):
        """Set the debug mode value."""
        self.debug_mode = value
