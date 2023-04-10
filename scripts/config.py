import abc
import json
import os
import openai
from urllib3.util import make_headers
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
            cls._instances[cls] = super(
                Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class AbstractSingleton(abc.ABC, metaclass=Singleton):
    pass


class Config(metaclass=Singleton):
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self):
        """Initialize the Config class"""
        self.debug = False
        self.continuous_mode = False
        self.speak_mode = False

        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo")
        self.smart_llm_model = os.getenv("SMART_LLM_MODEL", "gpt-4")
        self.fast_token_limit = int(os.getenv("FAST_TOKEN_LIMIT", 4000))
        self.smart_token_limit = int(os.getenv("SMART_TOKEN_LIMIT", 8000))

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_proxy = os.getenv("OPENAI_API_PROXY")
        try:
            self.openai_api_proxy = json.loads(self.openai_api_proxy)
        except ValueError:
            pass
        except NameError:
            pass
        except TypeError:
            pass
        self.use_azure = False
        self.use_azure = os.getenv("USE_AZURE") == 'True'
        if self.use_azure:
            self.openai_api_base = os.getenv("OPENAI_API_BASE")
            self.openai_api_version = os.getenv("OPENAI_API_VERSION")
            self.openai_deployment_id = os.getenv("OPENAI_DEPLOYMENT_ID")
            openai.api_type = "azure"
            openai.api_base = self.openai_api_base
            openai.api_version = self.openai_api_version

        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_api_proxy = os.getenv("ELEVENLABS_API_PROXY")
        try:
            self.elevenlabs_api_proxy = json.loads(self.elevenlabs_api_proxy)
        except ValueError:
            pass
        except NameError:
            pass
        except TypeError:
            pass

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_api_proxy = os.getenv("GOOGLE_API_PROXY")

        self.custom_search_engine_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_region = os.getenv("PINECONE_ENV")
        self.pinecone_proxy = os.getenv("PINECONE_PROXY")
        self.pinecone_proxy_headers = os.getenv("PINECONE_PROXY_HEADERS")
        self.pinecone_proxy_headers = make_headers(proxy_basic_auth=self.pinecone_proxy_headers)

        self.image_provider = os.getenv("IMAGE_PROVIDER")
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

        # User agent headers to use when browsing web
        # Some websites might just completely deny request with an error code if no user agent was found.
        self.user_agent_header = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = os.getenv("REDIS_PORT", "6379")
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.wipe_redis_on_start = os.getenv("WIPE_REDIS_ON_START", "True") == 'True'
        self.memory_index = os.getenv("MEMORY_INDEX", 'auto-gpt')
        # Note that indexes must be created on db 0 in redis, this is not configureable.

        self.memory_backend = os.getenv("MEMORY_BACKEND", 'local')
        # Initialize the OpenAI API client
        openai.api_key = self.openai_api_key
        openai.proxy = self.openai_api_proxy

    def set_continuous_mode(self, value: bool):
        """Set the continuous mode value."""
        self.continuous_mode = value

    def set_speak_mode(self, value: bool):
        """Set the speak mode value."""
        self.speak_mode = value

    def set_debug_mode(self, value: bool):
        self.debug_mode = value

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

    def set_openai_api_key(self, value: str):
        """Set the OpenAI API key value."""
        self.openai_api_key = value

    def set_openai_api_proxy(self, value: str | dict):
        """Set the OpenAI API proxy value."""
        self.openai_api_proxy = value

    def set_elevenlabs_api_key(self, value: str):
        """Set the ElevenLabs API key value."""
        self.elevenlabs_api_key = value

    def set_elevenlabs_api_proxy(self, value: str):
        """Set the ElevenLabs API proxy value."""
        self.elevenlabs_api_proxy = value

    def set_google_api_key(self, value: str):
        """Set the Google API key value."""
        self.google_api_key = value

    def set_google_api_proxy(self, value: str):
        """Set the Google API proxy value."""
        self.google_api_proxy = value

    def set_custom_search_engine_id(self, value: str):
        """Set the custom search engine id value."""
        self.custom_search_engine_id = value

    def set_pinecone_api_key(self, value: str):
        """Set the Pinecone API key value."""
        self.pinecone_api_key = value

    def set_pinecone_region(self, value: str):
        """Set the Pinecone region value."""
        self.pinecone_region = value

    def set_pinecone_proxy(self, value: str):
        """Set the Pinecone proxy value."""
        self.pinecone_proxy = value

    def set_pinecone_headers(self, value: str):
        """Set the Pinecone headers value."""
        self.pinecone_proxy_headers = make_headers(proxy_basic_auth=value)

    def set_debug_mode(self, value: bool):
        """Set the debug mode value."""
        self.debug = value
