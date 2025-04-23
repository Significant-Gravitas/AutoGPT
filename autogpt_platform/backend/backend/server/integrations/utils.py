from supabase import Client, create_client

from backend.util.settings import Settings

settings = Settings()


def get_supabase() -> Client:
    return create_client(
        settings.secrets.supabase_url, settings.secrets.supabase_service_role_key
    )
