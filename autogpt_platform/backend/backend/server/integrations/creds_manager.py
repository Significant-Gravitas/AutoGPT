import logging
from contextlib import contextmanager
from datetime import datetime

from autogpt_libs.supabase_integration_credentials_store import (
    Credentials,
    SupabaseIntegrationCredentialsStore,
)
from autogpt_libs.utils.synchronize import RedisKeyedMutex
from redis.lock import Lock as RedisLock

from backend.data import redis
from backend.integrations.oauth import HANDLERS_BY_NAME, BaseOAuthHandler
from backend.util.settings import Settings

from .utils import get_supabase

logger = logging.getLogger(__name__)
settings = Settings()


class IntegrationCredentialsManager:
    def __init__(self):
        redis_conn = redis.get_redis()
        self._locks = RedisKeyedMutex(redis_conn)
        self.store = SupabaseIntegrationCredentialsStore(get_supabase(), redis_conn)

    def create(self, user_id: str, credentials: Credentials) -> None:
        return self.store.add_creds(user_id, credentials)

    def exists(self, user_id: str, credentials_id: str) -> bool:
        return self.store.get_creds_by_id(user_id, credentials_id) is not None

    def get(
        self, user_id: str, credentials_id: str, lock: bool = True
    ) -> Credentials | None:
        credentials = self.store.get_creds_by_id(user_id, credentials_id)
        if not credentials:
            return None

        # Refresh OAuth credentials if needed
        if credentials.type == "oauth2" and credentials.access_token_expires_at:
            logger.debug(
                f"Credentials #{credentials.id} expire at "
                f"{datetime.fromtimestamp(credentials.access_token_expires_at)}; "
                f"current time is {datetime.now()}"
            )

            with self._locked(user_id, credentials_id, "refresh"):
                oauth_handler = _get_provider_oauth_handler(credentials.provider)
                if oauth_handler.needs_refresh(credentials):
                    logger.debug(
                        f"Refreshing '{credentials.provider}' "
                        f"credentials #{credentials.id}"
                    )
                    _lock = None
                    if lock:
                        # Wait until the credentials are no longer in use anywhere
                        _lock = self._acquire_lock(user_id, credentials_id)

                    fresh_credentials = oauth_handler.refresh_tokens(credentials)
                    self.store.update_creds(user_id, fresh_credentials)
                    if _lock:
                        _lock.release()

                    credentials = fresh_credentials
        else:
            logger.debug(f"Credentials #{credentials.id} never expire")

        return credentials

    def acquire(
        self, user_id: str, credentials_id: str
    ) -> tuple[Credentials, RedisLock]:
        # Use a low-priority (!time_sensitive) locking queue on top of the general lock
        # to allow priority access for refreshing/updating the tokens.
        with self._locked(user_id, credentials_id, "!time_sensitive"):
            lock = self._acquire_lock(user_id, credentials_id)
        credentials = self.get(user_id, credentials_id, lock=False)
        if not credentials:
            raise ValueError(
                f"Credentials #{credentials_id} for user #{user_id} not found"
            )
        return credentials, lock

    def update(self, user_id: str, updated: Credentials) -> None:
        with self._locked(user_id, updated.id):
            self.store.update_creds(user_id, updated)

    def delete(self, user_id: str, credentials_id: str) -> None:
        with self._locked(user_id, credentials_id):
            self.store.delete_creds_by_id(user_id, credentials_id)

    # -- Locking utilities -- #

    def _acquire_lock(self, user_id: str, credentials_id: str, *args: str) -> RedisLock:
        key = (self.store.supabase.supabase_url, user_id, credentials_id, *args)
        return self._locks.acquire(key)

    @contextmanager
    def _locked(self, user_id: str, credentials_id: str, *args: str):
        lock = self._acquire_lock(user_id, credentials_id, *args)
        try:
            yield
        finally:
            lock.release()

    def release_all_locks(self):
        """Call this on process termination to ensure all locks are released"""
        self._locks.release_all_locks()
        self.store.locks.release_all_locks()


def _get_provider_oauth_handler(provider_name: str) -> BaseOAuthHandler:
    if provider_name not in HANDLERS_BY_NAME:
        raise KeyError(f"Unknown provider '{provider_name}'")

    client_id = getattr(settings.secrets, f"{provider_name}_client_id")
    client_secret = getattr(settings.secrets, f"{provider_name}_client_secret")
    if not (client_id and client_secret):
        raise Exception(  # TODO: ConfigError
            f"Integration with provider '{provider_name}' is not configured",
        )

    handler_class = HANDLERS_BY_NAME[provider_name]
    frontend_base_url = settings.config.frontend_base_url
    return handler_class(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=f"{frontend_base_url}/auth/integrations/oauth_callback",
    )
