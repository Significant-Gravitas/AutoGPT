from enum import Enum
from typing import Any


# --8<-- [start:ProviderName]
class ProviderName(str, Enum):
    """
    Provider names for integrations.
    
    This enum extends str to accept any string value while maintaining
    backward compatibility with existing provider constants.
    """
    ANTHROPIC = "anthropic"
    APOLLO = "apollo"
    COMPASS = "compass"
    DISCORD = "discord"
    D_ID = "d_id"
    E2B = "e2b"
    EXA = "exa"
    FAL = "fal"
    GENERIC_WEBHOOK = "generic_webhook"
    GITHUB = "github"
    GOOGLE = "google"
    GOOGLE_MAPS = "google_maps"
    GROQ = "groq"
    HUBSPOT = "hubspot"
    IDEOGRAM = "ideogram"
    JINA = "jina"
    LINEAR = "linear"
    LLAMA_API = "llama_api"
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
    TWITTER = "twitter"
    TODOIST = "todoist"
    UNREAL_SPEECH = "unreal_speech"
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
    # --8<-- [end:ProviderName]
