from enum import Enum


class ProviderName(str, Enum):
    GITHUB = "github"
    GOOGLE = "google"
    NOTION = "notion"
