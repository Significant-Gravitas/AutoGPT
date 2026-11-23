"""The name people know a provider by, from its slug."""

# Where ``str.title()`` gets a provider's name wrong.
_PROVIDER_TITLES = {
    "github": "GitHub",
    "gitlab": "GitLab",
    "hubspot": "HubSpot",
    "linkedin": "LinkedIn",
    "openai": "OpenAI",
    "youtube": "YouTube",
}


def provider_title(provider: str) -> str:
    """``google`` is "Google", ``github`` "GitHub"."""
    return _PROVIDER_TITLES.get(provider, provider.replace("_", " ").title())
