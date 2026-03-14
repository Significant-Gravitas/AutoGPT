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
        # Google OAuth
        self.google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
        self.google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
        # Full callback URL — must match exactly what is registered in Google Cloud Console
        # Example: https://your-app.railway.app/auth/google/callback
        self.google_redirect_uri: str = os.getenv("GOOGLE_REDIRECT_URI", "")
        # WhatsApp Business Cloud API
        # App ID: set WHATSAPP_APP_ID (your Facebook App ID)
        # App Secret: set WHATSAPP_APP_SECRET — used to verify webhook signatures
        # Access Token: a permanent System User token from Meta Business Suite
        # Phone Number ID: found in WhatsApp → Getting Started in the Meta developer portal
        # Verify Token: any string you choose — must match what you enter in the webhook config UI
        self.whatsapp_app_id: str = os.getenv("WHATSAPP_APP_ID", "")
        self.whatsapp_app_secret: str = os.getenv("WHATSAPP_APP_SECRET", "")
        self.whatsapp_access_token: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
        self.whatsapp_phone_number_id: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
        self.whatsapp_verify_token: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")

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
