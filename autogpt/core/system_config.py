"""
Port of pervious configuration.
Uses pydantic to load all variables from .env.

"""
from typing import List, Optional

from pydantic import BaseSettings, Field


class SystemConfig(BaseSettings):
    """
    The system config loading all fields from env variables
    """

    # AUTO-GPT - GENERAL SETTINGS
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    execute_local_commands: bool = Field(False, env="EXECUTE_LOCAL_COMMANDS")
    restrict_to_workspace: bool = Field(True, env="RESTRICT_TO_WORKSPACE")
    user_agent: Optional[str] = Field(None, env="USER_AGENT")
    ai_settings_file: str = Field("ai_settings.yaml", env="AI_SETTINGS_FILE")
    plugins_config_file: str = Field("plugins_config.yaml", env="PLUGINS_CONFIG_FILE")
    prompt_settings_file: str = Field(
        "prompt_settings.yaml", env="PROMPT_SETTINGS_FILE"
    )
    openai_api_base_url: Optional[str] = Field(None, env="OPENAI_API_BASE_URL")
    openai_functions: bool = Field(False, env="OPENAI_FUNCTIONS")
    authorise_command_key: str = Field("y", env="AUTHORISE_COMMAND_KEY")
    exit_key: str = Field("n", env="EXIT_KEY")
    plain_output: bool = Field(False, env="PLAIN_OUTPUT")
    disabled_command_categories: Optional[List[str]] = Field(
        None, env="DISABLED_COMMAND_CATEGORIES"
    )

    # LLM PROVIDER
    temperature: int = Field(0, env="TEMPERATURE")
    openai_organization: Optional[str] = Field(None, env="OPENAI_ORGANIZATION")
    use_azure: bool = Field(False, env="USE_AZURE")

    # LLM MODELS
    smart_llm_model: str = Field("gpt-3.5-turbo", env="SMART_LLM_MODEL")
    fast_llm_model: str = Field("gpt-3.5-turbo", env="FAST_LLM_MODEL")
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")

    # SHELL EXECUTION
    shell_command_control: str = Field("denylist", env="SHELL_COMMAND_CONTROL")
    shell_denylist: List[str] = Field(["sudo", "su"], env="SHELL_DENYLIST")
    shell_allowlist: Optional[List[str]] = Field(None, env="SHELL_ALLOWLIST")

    # MEMORY
    memory_backend: str = Field("json_file", env="MEMORY_BACKEND")
    memory_index: str = Field("auto-gpt", env="MEMORY_INDEX")
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    wipe_redis_on_start: bool = Field(True, env="WIPE_REDIS_ON_START")

    # IMAGE GENERATION PROVIDER
    image_provider: str = Field("dalle", env="IMAGE_PROVIDER")
    image_size: int = Field(256, env="IMAGE_SIZE")
    huggingface_image_model: str = Field(
        "CompVis/stable-diffusion-v1-4", env="HUGGINGFACE_IMAGE_MODEL"
    )
    huggingface_api_token: Optional[str] = Field(None, env="HUGGINGFACE_API_TOKEN")
    sd_webui_auth: Optional[str] = Field(None, env="SD_WEBUI_AUTH")
    sd_webui_url: str = Field("http://localhost:7860", env="SD_WEBUI_URL")

    # AUDIO TO TEXT PROVIDER
    audio_to_text_provider: str = Field("huggingface", env="AUDIO_TO_TEXT_PROVIDER")
    huggingface_audio_to_text_model: str = Field(
        "facebook/wav2vec2-large-960h-lv60-self", env="HUGGINGFACE_AUDIO_TO_TEXT_MODEL"
    )

    # STABLE DIFFUSION
    sd_webui_auth: Optional[str] = Field(None, env="SD_WEBUI_AUTH")
    sd_webui_url: str = Field("http://localhost:7860", env="SD_WEBUI_URL")

    # GITHUB
    github_api_key: Optional[str] = Field(None, env="GITHUB_API_KEY")
    github_username: Optional[str] = Field(None, env="GITHUB_USERNAME")

    # WEB BROWSING
    headless_browser: bool = Field(True, env="HEADLESS_BROWSER")
    use_web_browser: str = Field("chrome", env="USE_WEB_BROWSER")
    browse_chunk_max_length: int = Field(3000, env="BROWSE_CHUNK_MAX_LENGTH")
    browse_spacy_language_model: str = Field(
        "en_core_web_sm", env="BROWSE_SPACY_LANGUAGE_MODEL"
    )
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    google_custom_search_engine_id: Optional[str] = Field(
        None, env="GOOGLE_CUSTOM_SEARCH_ENGINE_ID"
    )

    # TEXT TO SPEECH PROVIDER
    text_to_speech_provider: str = Field("gtts", env="TEXT_TO_SPEECH_PROVIDER")
    streamelements_voice: str = Field("Brian", env="STREAMELEMENTS_VOICE")
    elevenlabs_api_key: Optional[str] = Field(None, env="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: Optional[str] = Field(None, env="ELEVENLABS_VOICE_ID")

    # CHAT MESSAGES
    chat_messages_enabled: bool = Field(False, env="CHAT_MESSAGES_ENABLED")

    class Config:
        env_file = ".env"
