from autogpt_libs.supabase_integration_credentials_store import (
    SupabaseIntegrationCredentialsStore,
)
from fastapi import Depends
from supabase import Client, create_client

from backend.util.settings import Settings

settings = Settings()


def get_supabase() -> Client:
    return create_client(
        settings.secrets.supabase_url, settings.secrets.supabase_service_role_key
    )


def get_creds_store(supabase: Client = Depends(get_supabase)):
    return SupabaseIntegrationCredentialsStore(supabase)
