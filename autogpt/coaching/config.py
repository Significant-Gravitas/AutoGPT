"""Configuration for the ABN Consulting AI Co-Navigator."""
import os

from autogpt.singleton import Singleton


class CoachingConfig(metaclass=Singleton):
    """Reads coaching-specific settings from environment variables."""

    def __init__(self):
        self.coach_name: str = os.getenv("COACHING_COACH_NAME", "Adi Ben Nesher")
        self.coach_calendly_url: str = os.getenv(
            "COACHING_COACH_CALENDLY_URL", "https://calendly.com/abn_consulting/30min"
        )
        self.alert_red_threshold: int = int(os.getenv("COACHING_ALERT_RED_THRESHOLD", "25"))
        self.alert_yellow_threshold: int = int(os.getenv("COACHING_ALERT_YELLOW_THRESHOLD", "40"))
        self.api_key: str = os.getenv("COACHING_API_KEY", "")
        self.supabase_url: str = os.getenv("SUPABASE_URL", "")
        self.supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")
        # Claude LLM settings
        self.llm_model: str = os.getenv("COACHING_LLM_MODEL", "claude-haiku-4-5-20251001")
        self.llm_temperature: float = float(os.getenv("COACHING_LLM_TEMPERATURE", "0.7"))

    def validate(self) -> None:
        """Raise if required env vars are missing."""
        missing = []
        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        if not self.supabase_service_key:
            missing.append("SUPABASE_SERVICE_KEY")
        if not self.api_key:
            missing.append("COACHING_API_KEY")
        if not os.getenv("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


# Module-level singleton instance
coaching_config = CoachingConfig()
