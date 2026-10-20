DEFAULT_ON_FLAGS = frozenset(
    {
        "autogpt-new-layout",
        "chat-search",
        "artifacts-page",
        "hire-experts",
        "skills-hub",
        "onboarding-brain-dump",
        "onboarding-expert-team",
        "graphiti-memory",
        "graphiti-communities-enabled",
        "expert-spend-approval",
        "ai-agent-execution-summary",
    }
)


def boolean_flag_default(flag_key: str, default: bool | None = None) -> bool:
    """An explicit caller fallback wins over the selected product defaults."""
    return flag_key in DEFAULT_ON_FLAGS if default is None else default
