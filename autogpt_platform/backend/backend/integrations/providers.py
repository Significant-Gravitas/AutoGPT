from enum import Enum
from typing import Any


# --8<-- [start:ProviderName]
class ProviderName(str, Enum):
    """
    Provider names for integrations.

    This enum extends str to accept any string value while maintaining
    backward compatibility with existing provider constants.
    """

    AIML_API = "aiml_api"
    ANTHROPIC = "anthropic"
    APOLLO = "apollo"
    COMPASS = "compass"
    DISCORD = "discord"
    D_ID = "d_id"
    E2B = "e2b"
    ELEVENLABS = "elevenlabs"
    FAL = "fal"
    GITHUB = "github"
    GOOGLE = "google"
    GOOGLE_MAPS = "google_maps"
    GROQ = "groq"
    HTTP = "http"
    HUBSPOT = "hubspot"
    ENRICHLAYER = "enrichlayer"
    IDEOGRAM = "ideogram"
    JINA = "jina"
    LLAMA_API = "llama_api"
    MCP = "mcp"
    MEDIUM = "medium"
    MEM0 = "mem0"
    NOTION = "notion"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENWEATHERMAP = "openweathermap"
    OPEN_ROUTER = "open_router"
    PINECONE = "pinecone"
    REDDIT = "reddit"
    REPLICATE = "replicate"
    REVID = "revid"
    SCREENSHOTONE = "screenshotone"
    SLANT3D = "slant3d"
    SMARTLEAD = "smartlead"
    SMTP = "smtp"
    TELEGRAM = "telegram"
    TWITTER = "twitter"
    TODOIST = "todoist"
    UNREAL_SPEECH = "unreal_speech"
    V0 = "v0"
    WEBSHARE_PROXY = "webshare_proxy"
    ZEROBOUNCE = "zerobounce"

    @classmethod
    def _missing_(cls, value: Any) -> "ProviderName":
        """
        Allow any string value to be used as a ProviderName.
        This enables SDK users to define custom providers without
        modifying the enum.
        """
        if isinstance(value, str):
            # Create a pseudo-member that behaves like an enum member
            pseudo_member = str.__new__(cls, value)
            pseudo_member._name_ = value.upper()
            pseudo_member._value_ = value
            return pseudo_member
        return None  # type: ignore

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        """
        Custom JSON schema generation that allows any string value,
        not just the predefined enum values.
        """
        # Get the default schema
        json_schema = handler(schema)

        # Remove the enum constraint to allow any string
        if "enum" in json_schema:
            del json_schema["enum"]

        # Keep the type as string
        json_schema["type"] = "string"

        # Update description to indicate custom providers are allowed
        json_schema["description"] = (
            "Provider name for integrations. "
            "Can be any string value, including custom provider names."
        )

        return json_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """
        Pydantic v2 core schema that allows any string value.
        """
        from pydantic_core import core_schema

        # Create a string schema that validates any string
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
        )

    # --8<-- [end:ProviderName]
