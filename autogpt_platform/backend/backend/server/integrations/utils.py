from backend.util.settings import Settings
from supabase import Client, create_client

settings = Settings()


def get_supabase() -> Client:
    return create_client(
        settings.secrets.supabase_url, settings.secrets.supabase_service_role_key
    )
