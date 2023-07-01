"""Configuration class to store the state of bools for different scripts access."""
from __future__ import annotations

import contextlib
import os
import re
from typing import Dict

import yaml
from colorama import Fore

from autogpt.core.configuration.schema import Configurable, SystemSettings
from autogpt.plugins.plugins_config import PluginsConfig

AZURE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "../..", "azure.yaml")
from typing import Optional


class Config(SystemSettings):
    fast_llm_model: str
    smart_llm_model: str
    continuous_mode: bool
    skip_news: bool
    workspace_path: Optional[str] = None
    file_logger_path: Optional[str] = None
    debug_mode: bool
    plugins_dir: str
    plugins_config: PluginsConfig
    continuous_limit: int
    speak_mode: bool
    skip_reprompt: bool
    allow_downloads: bool
    exit_key: str
    plain_output: bool
    disabled_command_categories: list[str]
    shell_command_control: str
    shell_denylist: list[str]
    shell_allowlist: list[str]
    ai_settings_file: str
    prompt_settings_file: str
    embedding_model: str
    browse_spacy_language_model: str
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    temperature: float
    use_azure: bool
    execute_local_commands: bool
    restrict_to_workspace: bool
    openai_api_type: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None
    openai_functions: bool
    elevenlabs_api_key: Optional[str] = None
    streamelements_voice: str
    text_to_speech_provider: str
    github_api_key: Optional[str] = None
    github_username: Optional[str] = None
    google_api_key: Optional[str] = None
    google_custom_search_engine_id: Optional[str] = None
    image_provider: Optional[str] = None
    image_size: int
    huggingface_api_token: Optional[str] = None
    huggingface_image_model: str
    audio_to_text_provider: str
    huggingface_audio_to_text_model: Optional[str] = None
    sd_webui_url: Optional[str] = None
    sd_webui_auth: Optional[str] = None
    selenium_web_browser: str
    selenium_headless: bool
    user_agent: str
    memory_backend: str
    memory_index: str
    redis_host: str
    redis_port: int
    redis_password: str
    wipe_redis_on_start: bool
    plugins_allowlist: list[str]
    plugins_denylist: list[str]
    plugins_openai: list[str]
    plugins_config_file: str
    chat_messages_enabled: bool
    elevenlabs_voice_id: Optional[str] = None
    plugins: list[str]
    authorise_key: str

    # Executed immediately after init by Pydantic
    def model_post_init(self, **kwargs) -> None:
        if not self.plugins_config.plugins:
            self.plugins_config = PluginsConfig.load_config(self)


class ConfigBuilder(Configurable[Config]):
    default_plugins_config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugins_config.yaml"
    )

    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if os.getenv("USE_MAC_OS_TTS"):
        default_tts_provider = "macos"
    elif elevenlabs_api_key:
        default_tts_provider = "elevenlabs"
    elif os.getenv("USE_BRIAN_TTS"):
        default_tts_provider = "streamelements"
    else:
        default_tts_provider = "gtts"

    defaults_settings = Config(
        name="Default Server Config",
        description="This is a default server configuration",
        smart_llm_model="gpt-3.5-turbo",
        fast_llm_model="gpt-3.5-turbo",
        continuous_mode=False,
        continuous_limit=0,
        skip_news=False,
        debug_mode=False,
        plugins_dir="plugins",
        plugins_config=PluginsConfig({}),
        speak_mode=False,
        skip_reprompt=False,
        allow_downloads=False,
        exit_key="n",
        plain_output=False,
        disabled_command_categories=[],
        shell_command_control="denylist",
        shell_denylist=["sudo", "su"],
        shell_allowlist=[],
        ai_settings_file="ai_settings.yaml",
        prompt_settings_file="prompt_settings.yaml",
        embedding_model="text-embedding-ada-002",
        browse_spacy_language_model="en_core_web_sm",
        temperature=0,
        use_azure=False,
        execute_local_commands=False,
        restrict_to_workspace=True,
        openai_functions=False,
        streamelements_voice="Brian",
        text_to_speech_provider=default_tts_provider,
        image_size=256,
        huggingface_image_model="CompVis/stable-diffusion-v1-4",
        audio_to_text_provider="huggingface",
        sd_webui_url="http://localhost:7860",
        selenium_web_browser="chrome",
        selenium_headless=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
        memory_backend="json_file",
        memory_index="auto-gpt-memory",
        redis_host="localhost",
        redis_port=6379,
        wipe_redis_on_start=True,
        plugins_allowlist=[],
        plugins_denylist=[],
        plugins_openai=[],
        plugins_config_file=default_plugins_config_file,
        chat_messages_enabled=True,
        plugins=[],
        authorise_key="y",
        redis_password="",
    )

    @classmethod
    def build_config_from_env(cls):
        """Initialize the Config class"""
        config_dict = {
            "authorise_key": os.getenv("AUTHORISE_COMMAND_KEY"),
            "exit_key": os.getenv("EXIT_KEY"),
            "plain_output": os.getenv("PLAIN_OUTPUT", "False") == "True",
            "shell_command_control": os.getenv("SHELL_COMMAND_CONTROL"),
            "ai_settings_file": os.getenv("AI_SETTINGS_FILE"),
            "prompt_settings_file": os.getenv("PROMPT_SETTINGS_FILE"),
            "fast_llm_model": os.getenv("FAST_LLM_MODEL"),
            "smart_llm_model": os.getenv("SMART_LLM_MODEL"),
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
            "browse_spacy_language_model": os.getenv("BROWSE_SPACY_LANGUAGE_MODEL"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "use_azure": os.getenv("USE_AZURE") == "True",
            "execute_local_commands": os.getenv("EXECUTE_LOCAL_COMMANDS", "False")
            == "True",
            "restrict_to_workspace": os.getenv("RESTRICT_TO_WORKSPACE", "True")
            == "True",
            "openai_functions": os.getenv("OPENAI_FUNCTIONS", "False") == "True",
            "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY"),
            "streamelements_voice": os.getenv("STREAMELEMENTS_VOICE"),
            "text_to_speech_provider": os.getenv("TEXT_TO_SPEECH_PROVIDER"),
            "github_api_key": os.getenv("GITHUB_API_KEY"),
            "github_username": os.getenv("GITHUB_USERNAME"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "image_provider": os.getenv("IMAGE_PROVIDER"),
            "huggingface_api_token": os.getenv("HUGGINGFACE_API_TOKEN"),
            "huggingface_image_model": os.getenv("HUGGINGFACE_IMAGE_MODEL"),
            "audio_to_text_provider": os.getenv("AUDIO_TO_TEXT_PROVIDER"),
            "huggingface_audio_to_text_model": os.getenv(
                "HUGGINGFACE_AUDIO_TO_TEXT_MODEL"
            ),
            "sd_webui_url": os.getenv("SD_WEBUI_URL"),
            "sd_webui_auth": os.getenv("SD_WEBUI_AUTH"),
            "selenium_web_browser": os.getenv("USE_WEB_BROWSER"),
            "selenium_headless": os.getenv("HEADLESS_BROWSER", "True") == "True",
            "user_agent": os.getenv("USER_AGENT"),
            "memory_backend": os.getenv("MEMORY_BACKEND"),
            "memory_index": os.getenv("MEMORY_INDEX"),
            "redis_host": os.getenv("REDIS_HOST"),
            "redis_password": os.getenv("REDIS_PASSWORD"),
            "wipe_redis_on_start": os.getenv("WIPE_REDIS_ON_START", "True") == "True",
            "plugins_dir": os.getenv("PLUGINS_DIR"),
            "plugins_config_file": os.getenv("PLUGINS_CONFIG_FILE"),
            "chat_messages_enabled": os.getenv("CHAT_MESSAGES_ENABLED") == "True",
        }

        # Converting to a list from comma-separated string
        disabled_command_categories = os.getenv("DISABLED_COMMAND_CATEGORIES")
        if disabled_command_categories:
            config_dict[
                "disabled_command_categories"
            ] = disabled_command_categories.split(",")

        # Converting to a list from comma-separated string
        shell_denylist = os.getenv("SHELL_DENYLIST", os.getenv("DENY_COMMANDS"))
        if shell_denylist:
            config_dict["shell_denylist"] = shell_denylist.split(",")

        shell_allowlist = os.getenv("SHELL_ALLOWLIST", os.getenv("ALLOW_COMMANDS"))
        if shell_allowlist:
            config_dict["shell_allowlist"] = shell_allowlist.split(",")

        config_dict["google_custom_search_engine_id"] = os.getenv(
            "GOOGLE_CUSTOM_SEARCH_ENGINE_ID", os.getenv("CUSTOM_SEARCH_ENGINE_ID")
        )

        config_dict["elevenlabs_voice_id"] = os.getenv(
            "ELEVENLABS_VOICE_ID", os.getenv("ELEVENLABS_VOICE_1_ID")
        )

        plugins_allowlist = os.getenv("ALLOWLISTED_PLUGINS")
        if plugins_allowlist:
            config_dict["plugins_allowlist"] = plugins_allowlist.split(",")

        plugins_denylist = os.getenv("DENYLISTED_PLUGINS")
        if plugins_denylist:
            config_dict["plugins_denylist"] = plugins_denylist.split(",")

        with contextlib.suppress(TypeError):
            config_dict["image_size"] = int(os.getenv("IMAGE_SIZE"))
        with contextlib.suppress(TypeError):
            config_dict["redis_port"] = int(os.getenv("REDIS_PORT"))
        with contextlib.suppress(TypeError):
            config_dict["temperature"] = float(os.getenv("TEMPERATURE"))

        if config_dict["use_azure"]:
            azure_config = cls.load_azure_config()
            config_dict["openai_api_type"] = azure_config["openai_api_type"]
            config_dict["openai_api_base"] = azure_config["openai_api_base"]
            config_dict["openai_api_version"] = azure_config["openai_api_version"]

        if os.getenv("OPENAI_API_BASE_URL"):
            config_dict["openai_api_base"] = os.getenv("OPENAI_API_BASE_URL")

        openai_organization = os.getenv("OPENAI_ORGANIZATION")
        if openai_organization is not None:
            config_dict["openai_organization"] = openai_organization

        config_dict_without_none_values = {
            k: v for k, v in config_dict.items() if v is not None
        }

        return cls.build_agent_configuration(config_dict_without_none_values)

    @classmethod
    def load_azure_config(cls, config_file: str = AZURE_CONFIG_FILE) -> Dict[str, str]:
        """
        Loads the configuration parameters for Azure hosting from the specified file
          path as a yaml file.

        Parameters:
            config_file(str): The path to the config yaml file. DEFAULT: "../azure.yaml"

        Returns:
            Dict
        """
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader) or {}

        return {
            "openai_api_type": config_params.get("azure_api_type") or "azure",
            "openai_api_base": config_params.get("azure_api_base") or "",
            "openai_api_version": config_params.get("azure_api_version")
            or "2023-03-15-preview",
            "azure_model_to_deployment_id_map": config_params.get(
                "azure_model_map", {}
            ),
        }


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
            config.openai_api_key = openai_api_key
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
