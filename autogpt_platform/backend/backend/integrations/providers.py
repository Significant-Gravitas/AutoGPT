from enum import Enum


# --8<-- [start:ProviderName]
class ProviderName(str, Enum):
    ANTHROPIC = "anthropic"
    COMPASS = "compass"
    DISCORD = "discord"
    D_ID = "d_id"
    E2B = "e2b"
    EXA = "exa"
    FAL = "fal"
    GITHUB = "github"
    GOOGLE = "google"
    GOOGLE_MAPS = "google_maps"
    GROQ = "groq"
    HUBSPOT = "hubspot"
    IDEOGRAM = "ideogram"
    JINA = "jina"
    MEDIUM = "medium"
    NOTION = "notion"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENWEATHERMAP = "openweathermap"
    OPEN_ROUTER = "open_router"
    PINECONE = "pinecone"
    REPLICATE = "replicate"
    REVID = "revid"
    SLANT3D = "slant3d"
    TWITTER = "twitter"
    UNREAL_SPEECH = "unreal_speech"
    # --8<-- [end:ProviderName]
