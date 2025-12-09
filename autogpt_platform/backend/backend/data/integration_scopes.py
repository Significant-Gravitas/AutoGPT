"""
Integration scopes mapping.

Maps AutoGPT's fine-grained integration scopes to provider-specific OAuth scopes.
These scopes are used to request granular permissions when connecting integrations
through the Credential Broker.
"""

from enum import Enum
from typing import Optional

from backend.integrations.providers import ProviderName


class IntegrationScope(str, Enum):
    """
    Fine-grained integration scopes for credential grants.

    Format: {provider}:{resource}.{permission}
    """

    # Google scopes
    GOOGLE_EMAIL_READ = "google:email.read"
    GOOGLE_GMAIL_READONLY = "google:gmail.readonly"
    GOOGLE_GMAIL_SEND = "google:gmail.send"
    GOOGLE_GMAIL_MODIFY = "google:gmail.modify"
    GOOGLE_DRIVE_READONLY = "google:drive.readonly"
    GOOGLE_DRIVE_FILE = "google:drive.file"
    GOOGLE_CALENDAR_READONLY = "google:calendar.readonly"
    GOOGLE_CALENDAR_EVENTS = "google:calendar.events"
    GOOGLE_SHEETS_READONLY = "google:sheets.readonly"
    GOOGLE_SHEETS = "google:sheets"
    GOOGLE_DOCS_READONLY = "google:docs.readonly"
    GOOGLE_DOCS = "google:docs"

    # GitHub scopes
    GITHUB_REPOS_READ = "github:repos.read"
    GITHUB_REPOS_WRITE = "github:repos.write"
    GITHUB_ISSUES_READ = "github:issues.read"
    GITHUB_ISSUES_WRITE = "github:issues.write"
    GITHUB_USER_READ = "github:user.read"
    GITHUB_GISTS = "github:gists"
    GITHUB_NOTIFICATIONS = "github:notifications"

    # Discord scopes
    DISCORD_IDENTIFY = "discord:identify"
    DISCORD_EMAIL = "discord:email"
    DISCORD_GUILDS = "discord:guilds"
    DISCORD_MESSAGES_READ = "discord:messages.read"

    # Twitter scopes
    TWITTER_READ = "twitter:read"
    TWITTER_WRITE = "twitter:write"
    TWITTER_DM = "twitter:dm"

    # Notion scopes
    NOTION_READ = "notion:read"
    NOTION_WRITE = "notion:write"

    # Todoist scopes
    TODOIST_READ = "todoist:read"
    TODOIST_WRITE = "todoist:write"


# Scope descriptions for consent UI
INTEGRATION_SCOPE_DESCRIPTIONS: dict[str, str] = {
    # Google
    IntegrationScope.GOOGLE_EMAIL_READ.value: "Read your email address",
    IntegrationScope.GOOGLE_GMAIL_READONLY.value: "Read your Gmail messages",
    IntegrationScope.GOOGLE_GMAIL_SEND.value: "Send emails on your behalf",
    IntegrationScope.GOOGLE_GMAIL_MODIFY.value: "Read, send, and manage your emails",
    IntegrationScope.GOOGLE_DRIVE_READONLY.value: "View files in your Google Drive",
    IntegrationScope.GOOGLE_DRIVE_FILE.value: "Create and edit files in Google Drive",
    IntegrationScope.GOOGLE_CALENDAR_READONLY.value: "View your calendar",
    IntegrationScope.GOOGLE_CALENDAR_EVENTS.value: "Create and edit calendar events",
    IntegrationScope.GOOGLE_SHEETS_READONLY.value: "View your spreadsheets",
    IntegrationScope.GOOGLE_SHEETS.value: "Create and edit spreadsheets",
    IntegrationScope.GOOGLE_DOCS_READONLY.value: "View your documents",
    IntegrationScope.GOOGLE_DOCS.value: "Create and edit documents",
    # GitHub
    IntegrationScope.GITHUB_REPOS_READ.value: "Read repository information",
    IntegrationScope.GITHUB_REPOS_WRITE.value: "Create and manage repositories",
    IntegrationScope.GITHUB_ISSUES_READ.value: "Read issues and pull requests",
    IntegrationScope.GITHUB_ISSUES_WRITE.value: "Create and manage issues",
    IntegrationScope.GITHUB_USER_READ.value: "Read your GitHub profile",
    IntegrationScope.GITHUB_GISTS.value: "Create and manage gists",
    IntegrationScope.GITHUB_NOTIFICATIONS.value: "Access notifications",
    # Discord
    IntegrationScope.DISCORD_IDENTIFY.value: "Access your Discord username",
    IntegrationScope.DISCORD_EMAIL.value: "Access your Discord email",
    IntegrationScope.DISCORD_GUILDS.value: "View your server list",
    IntegrationScope.DISCORD_MESSAGES_READ.value: "Read messages",
    # Twitter
    IntegrationScope.TWITTER_READ.value: "Read tweets and profile",
    IntegrationScope.TWITTER_WRITE.value: "Post tweets on your behalf",
    IntegrationScope.TWITTER_DM.value: "Send and read direct messages",
    # Notion
    IntegrationScope.NOTION_READ.value: "View Notion pages",
    IntegrationScope.NOTION_WRITE.value: "Create and edit Notion pages",
    # Todoist
    IntegrationScope.TODOIST_READ.value: "View your tasks",
    IntegrationScope.TODOIST_WRITE.value: "Create and manage tasks",
}


# Mapping from integration scopes to provider OAuth scopes
INTEGRATION_SCOPE_MAPPING: dict[str, dict[str, list[str]]] = {
    ProviderName.GOOGLE.value: {
        IntegrationScope.GOOGLE_EMAIL_READ.value: [
            "https://www.googleapis.com/auth/userinfo.email",
            "openid",
        ],
        IntegrationScope.GOOGLE_GMAIL_READONLY.value: [
            "https://www.googleapis.com/auth/gmail.readonly",
        ],
        IntegrationScope.GOOGLE_GMAIL_SEND.value: [
            "https://www.googleapis.com/auth/gmail.send",
        ],
        IntegrationScope.GOOGLE_GMAIL_MODIFY.value: [
            "https://www.googleapis.com/auth/gmail.modify",
        ],
        IntegrationScope.GOOGLE_DRIVE_READONLY.value: [
            "https://www.googleapis.com/auth/drive.readonly",
        ],
        IntegrationScope.GOOGLE_DRIVE_FILE.value: [
            "https://www.googleapis.com/auth/drive.file",
        ],
        IntegrationScope.GOOGLE_CALENDAR_READONLY.value: [
            "https://www.googleapis.com/auth/calendar.readonly",
        ],
        IntegrationScope.GOOGLE_CALENDAR_EVENTS.value: [
            "https://www.googleapis.com/auth/calendar.events",
        ],
        IntegrationScope.GOOGLE_SHEETS_READONLY.value: [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
        ],
        IntegrationScope.GOOGLE_SHEETS.value: [
            "https://www.googleapis.com/auth/spreadsheets",
        ],
        IntegrationScope.GOOGLE_DOCS_READONLY.value: [
            "https://www.googleapis.com/auth/documents.readonly",
        ],
        IntegrationScope.GOOGLE_DOCS.value: [
            "https://www.googleapis.com/auth/documents",
        ],
    },
    ProviderName.GITHUB.value: {
        IntegrationScope.GITHUB_REPOS_READ.value: [
            "repo:status",
            "public_repo",
        ],
        IntegrationScope.GITHUB_REPOS_WRITE.value: [
            "repo",
        ],
        IntegrationScope.GITHUB_ISSUES_READ.value: [
            "repo:status",
        ],
        IntegrationScope.GITHUB_ISSUES_WRITE.value: [
            "repo",
        ],
        IntegrationScope.GITHUB_USER_READ.value: [
            "read:user",
            "user:email",
        ],
        IntegrationScope.GITHUB_GISTS.value: [
            "gist",
        ],
        IntegrationScope.GITHUB_NOTIFICATIONS.value: [
            "notifications",
        ],
    },
    ProviderName.DISCORD.value: {
        IntegrationScope.DISCORD_IDENTIFY.value: [
            "identify",
        ],
        IntegrationScope.DISCORD_EMAIL.value: [
            "email",
        ],
        IntegrationScope.DISCORD_GUILDS.value: [
            "guilds",
        ],
        IntegrationScope.DISCORD_MESSAGES_READ.value: [
            "messages.read",
        ],
    },
    ProviderName.TWITTER.value: {
        IntegrationScope.TWITTER_READ.value: [
            "tweet.read",
            "users.read",
        ],
        IntegrationScope.TWITTER_WRITE.value: [
            "tweet.write",
        ],
        IntegrationScope.TWITTER_DM.value: [
            "dm.read",
            "dm.write",
        ],
    },
    ProviderName.NOTION.value: {
        IntegrationScope.NOTION_READ.value: [],  # Notion uses workspace-level access
        IntegrationScope.NOTION_WRITE.value: [],
    },
    ProviderName.TODOIST.value: {
        IntegrationScope.TODOIST_READ.value: [
            "data:read",
        ],
        IntegrationScope.TODOIST_WRITE.value: [
            "data:read_write",
        ],
    },
}


def get_provider_scopes(
    provider: ProviderName | str, integration_scopes: list[str]
) -> list[str]:
    """
    Convert integration scopes to provider-specific OAuth scopes.

    Args:
        provider: The provider name
        integration_scopes: List of integration scope strings

    Returns:
        List of provider-specific OAuth scopes
    """
    provider_value = provider.value if isinstance(provider, ProviderName) else provider
    provider_mapping = INTEGRATION_SCOPE_MAPPING.get(provider_value, {})

    oauth_scopes: set[str] = set()
    for scope in integration_scopes:
        if scope in provider_mapping:
            oauth_scopes.update(provider_mapping[scope])

    return list(oauth_scopes)


def get_provider_for_scope(scope: str) -> Optional[ProviderName]:
    """
    Get the provider for an integration scope.

    Args:
        scope: Integration scope string (e.g., "google:gmail.readonly")

    Returns:
        ProviderName or None if not recognized
    """
    if ":" not in scope:
        return None

    provider_prefix = scope.split(":")[0]

    # Map prefixes to providers
    prefix_mapping = {
        "google": ProviderName.GOOGLE,
        "github": ProviderName.GITHUB,
        "discord": ProviderName.DISCORD,
        "twitter": ProviderName.TWITTER,
        "notion": ProviderName.NOTION,
        "todoist": ProviderName.TODOIST,
    }

    return prefix_mapping.get(provider_prefix)


def validate_integration_scopes(scopes: list[str]) -> tuple[bool, list[str]]:
    """
    Validate a list of integration scopes.

    Args:
        scopes: List of integration scope strings

    Returns:
        Tuple of (valid, invalid_scopes)
    """
    valid_scopes = {s.value for s in IntegrationScope}
    invalid = [s for s in scopes if s not in valid_scopes]
    return len(invalid) == 0, invalid


def group_scopes_by_provider(
    scopes: list[str],
) -> dict[ProviderName, list[str]]:
    """
    Group integration scopes by their provider.

    Args:
        scopes: List of integration scope strings

    Returns:
        Dictionary mapping providers to their scopes
    """
    grouped: dict[ProviderName, list[str]] = {}

    for scope in scopes:
        provider = get_provider_for_scope(scope)
        if provider:
            if provider not in grouped:
                grouped[provider] = []
            grouped[provider].append(scope)

    return grouped
