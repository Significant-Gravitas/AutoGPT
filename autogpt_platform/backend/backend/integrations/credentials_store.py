import base64
import hashlib
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from autogpt_libs.utils.cache import thread_cached
from autogpt_libs.utils.synchronize import AsyncRedisKeyedMutex
from pydantic import SecretStr

from backend.data.db import prisma
from backend.data.model import (
    APIKeyCredentials,
    Credentials,
    OAuth2Credentials,
    OAuthState,
    UserIntegrations,
)
from backend.data.redis_client import get_redis_async
from backend.util.settings import Settings

settings = Settings()

# This is an overrride since ollama doesn't actually require an API key, but the creddential system enforces one be attached
ollama_credentials = APIKeyCredentials(
    id="744fdc56-071a-4761-b5a5-0af0ce10a2b5",
    provider="ollama",
    api_key=SecretStr("FAKE_API_KEY"),
    title="Use Credits for Ollama",
    expires_at=None,
)

revid_credentials = APIKeyCredentials(
    id="fdb7f412-f519-48d1-9b5f-d2f73d0e01fe",
    provider="revid",
    api_key=SecretStr(settings.secrets.revid_api_key),
    title="Use Credits for Revid",
    expires_at=None,
)
ideogram_credentials = APIKeyCredentials(
    id="760f84fc-b270-42de-91f6-08efe1b512d0",
    provider="ideogram",
    api_key=SecretStr(settings.secrets.ideogram_api_key),
    title="Use Credits for Ideogram",
    expires_at=None,
)
replicate_credentials = APIKeyCredentials(
    id="6b9fc200-4726-4973-86c9-cd526f5ce5db",
    provider="replicate",
    api_key=SecretStr(settings.secrets.replicate_api_key),
    title="Use Credits for Replicate",
    expires_at=None,
)
openai_credentials = APIKeyCredentials(
    id="53c25cb8-e3ee-465c-a4d1-e75a4c899c2a",
    provider="openai",
    api_key=SecretStr(settings.secrets.openai_api_key),
    title="Use Credits for OpenAI",
    expires_at=None,
)
aiml_api_credentials = APIKeyCredentials(
    id="aad82a89-9794-4ebb-977f-d736aa5260a3",
    provider="aiml_api",
    api_key=SecretStr(settings.secrets.aiml_api_key),
    title="Use Credits for AI/ML API",
    expires_at=None,
)
anthropic_credentials = APIKeyCredentials(
    id="24e5d942-d9e3-4798-8151-90143ee55629",
    provider="anthropic",
    api_key=SecretStr(settings.secrets.anthropic_api_key),
    title="Use Credits for Anthropic",
    expires_at=None,
)
groq_credentials = APIKeyCredentials(
    id="4ec22295-8f97-4dd1-b42b-2c6957a02545",
    provider="groq",
    api_key=SecretStr(settings.secrets.groq_api_key),
    title="Use Credits for Groq",
    expires_at=None,
)
did_credentials = APIKeyCredentials(
    id="7f7b0654-c36b-4565-8fa7-9a52575dfae2",
    provider="d_id",
    api_key=SecretStr(settings.secrets.did_api_key),
    title="Use Credits for D-ID",
    expires_at=None,
)
jina_credentials = APIKeyCredentials(
    id="7f26de70-ba0d-494e-ba76-238e65e7b45f",
    provider="jina",
    api_key=SecretStr(settings.secrets.jina_api_key),
    title="Use Credits for Jina",
    expires_at=None,
)
unreal_credentials = APIKeyCredentials(
    id="66f20754-1b81-48e4-91d0-f4f0dd82145f",
    provider="unreal",
    api_key=SecretStr(settings.secrets.unreal_speech_api_key),
    title="Use Credits for Unreal",
    expires_at=None,
)
open_router_credentials = APIKeyCredentials(
    id="b5a0e27d-0c98-4df3-a4b9-10193e1f3c40",
    provider="open_router",
    api_key=SecretStr(settings.secrets.open_router_api_key),
    title="Use Credits for Open Router",
    expires_at=None,
)
fal_credentials = APIKeyCredentials(
    id="6c0f5bd0-9008-4638-9d79-4b40b631803e",
    provider="fal",
    api_key=SecretStr(settings.secrets.fal_api_key),
    title="Use Credits for FAL",
    expires_at=None,
)
exa_credentials = APIKeyCredentials(
    id="96153e04-9c6c-4486-895f-5bb683b1ecec",
    provider="exa",
    api_key=SecretStr(settings.secrets.exa_api_key),
    title="Use Credits for Exa search",
    expires_at=None,
)
e2b_credentials = APIKeyCredentials(
    id="78d19fd7-4d59-4a16-8277-3ce310acf2b7",
    provider="e2b",
    api_key=SecretStr(settings.secrets.e2b_api_key),
    title="Use Credits for E2B",
    expires_at=None,
)
nvidia_credentials = APIKeyCredentials(
    id="96b83908-2789-4dec-9968-18f0ece4ceb3",
    provider="nvidia",
    api_key=SecretStr(settings.secrets.nvidia_api_key),
    title="Use Credits for Nvidia",
    expires_at=None,
)
screenshotone_credentials = APIKeyCredentials(
    id="3b1bdd16-8818-4bc2-8cbb-b23f9a3439ed",
    provider="screenshotone",
    api_key=SecretStr(settings.secrets.screenshotone_api_key),
    title="Use Credits for ScreenshotOne",
    expires_at=None,
)
mem0_credentials = APIKeyCredentials(
    id="ed55ac19-356e-4243-a6cb-bc599e9b716f",
    provider="mem0",
    api_key=SecretStr(settings.secrets.mem0_api_key),
    title="Use Credits for Mem0",
    expires_at=None,
)

apollo_credentials = APIKeyCredentials(
    id="544c62b5-1d0f-4156-8fb4-9525f11656eb",
    provider="apollo",
    api_key=SecretStr(settings.secrets.apollo_api_key),
    title="Use Credits for Apollo",
    expires_at=None,
)

smartlead_credentials = APIKeyCredentials(
    id="3bcdbda3-84a3-46af-8fdb-bfd2472298b8",
    provider="smartlead",
    api_key=SecretStr(settings.secrets.smartlead_api_key),
    title="Use Credits for SmartLead",
    expires_at=None,
)

google_maps_credentials = APIKeyCredentials(
    id="9aa1bde0-4947-4a70-a20c-84daa3850d52",
    provider="google_maps",
    api_key=SecretStr(settings.secrets.google_maps_api_key),
    title="Use Credits for Google Maps",
    expires_at=None,
)

zerobounce_credentials = APIKeyCredentials(
    id="63a6e279-2dc2-448e-bf57-85776f7176dc",
    provider="zerobounce",
    api_key=SecretStr(settings.secrets.zerobounce_api_key),
    title="Use Credits for ZeroBounce",
    expires_at=None,
)

llama_api_credentials = APIKeyCredentials(
    id="d44045af-1c33-4833-9e19-752313214de2",
    provider="llama_api",
    api_key=SecretStr(settings.secrets.llama_api_key),
    title="Use Credits for Llama API",
    expires_at=None,
)

DEFAULT_CREDENTIALS = [
    ollama_credentials,
    revid_credentials,
    ideogram_credentials,
    replicate_credentials,
    openai_credentials,
    aiml_api_credentials,
    anthropic_credentials,
    groq_credentials,
    did_credentials,
    jina_credentials,
    unreal_credentials,
    open_router_credentials,
    fal_credentials,
    exa_credentials,
    e2b_credentials,
    mem0_credentials,
    nvidia_credentials,
    screenshotone_credentials,
    apollo_credentials,
    smartlead_credentials,
    zerobounce_credentials,
    google_maps_credentials,
]


class IntegrationCredentialsStore:
    def __init__(self):
        self._locks = None

    async def locks(self) -> AsyncRedisKeyedMutex:
        if self._locks:
            return self._locks

        self._locks = AsyncRedisKeyedMutex(await get_redis_async())
        return self._locks

    @property
    @thread_cached
    def db_manager(self):
        if prisma.is_connected():
            from backend.data import user

            return user
        else:
            from backend.executor.database import DatabaseManagerAsyncClient
            from backend.util.service import get_service_client

            return get_service_client(DatabaseManagerAsyncClient)

    # =============== USER-MANAGED CREDENTIALS =============== #
    async def add_creds(self, user_id: str, credentials: Credentials) -> None:
        async with await self.locked_user_integrations(user_id):
            if await self.get_creds_by_id(user_id, credentials.id):
                raise ValueError(
                    f"Can not re-create existing credentials #{credentials.id} "
                    f"for user #{user_id}"
                )
            await self._set_user_integration_creds(
                user_id, [*(await self.get_all_creds(user_id)), credentials]
            )

    async def get_all_creds(self, user_id: str) -> list[Credentials]:
        users_credentials = (await self._get_user_integrations(user_id)).credentials
        all_credentials = users_credentials
        # These will always be added
        all_credentials.append(ollama_credentials)

        # These will only be added if the API key is set
        if settings.secrets.revid_api_key:
            all_credentials.append(revid_credentials)
        if settings.secrets.ideogram_api_key:
            all_credentials.append(ideogram_credentials)
        if settings.secrets.groq_api_key:
            all_credentials.append(groq_credentials)
        if settings.secrets.replicate_api_key:
            all_credentials.append(replicate_credentials)
        if settings.secrets.openai_api_key:
            all_credentials.append(openai_credentials)
        if settings.secrets.aiml_api_key:
            all_credentials.append(aiml_api_credentials)
        if settings.secrets.anthropic_api_key:
            all_credentials.append(anthropic_credentials)
        if settings.secrets.did_api_key:
            all_credentials.append(did_credentials)
        if settings.secrets.jina_api_key:
            all_credentials.append(jina_credentials)
        if settings.secrets.unreal_speech_api_key:
            all_credentials.append(unreal_credentials)
        if settings.secrets.open_router_api_key:
            all_credentials.append(open_router_credentials)
        if settings.secrets.fal_api_key:
            all_credentials.append(fal_credentials)
        if settings.secrets.exa_api_key:
            all_credentials.append(exa_credentials)
        if settings.secrets.e2b_api_key:
            all_credentials.append(e2b_credentials)
        if settings.secrets.nvidia_api_key:
            all_credentials.append(nvidia_credentials)
        if settings.secrets.screenshotone_api_key:
            all_credentials.append(screenshotone_credentials)
        if settings.secrets.mem0_api_key:
            all_credentials.append(mem0_credentials)
        if settings.secrets.apollo_api_key:
            all_credentials.append(apollo_credentials)
        if settings.secrets.smartlead_api_key:
            all_credentials.append(smartlead_credentials)
        if settings.secrets.zerobounce_api_key:
            all_credentials.append(zerobounce_credentials)
        if settings.secrets.google_maps_api_key:
            all_credentials.append(google_maps_credentials)
        return all_credentials

    async def get_creds_by_id(
        self, user_id: str, credentials_id: str
    ) -> Credentials | None:
        all_credentials = await self.get_all_creds(user_id)
        return next((c for c in all_credentials if c.id == credentials_id), None)

    async def get_creds_by_provider(
        self, user_id: str, provider: str
    ) -> list[Credentials]:
        credentials = await self.get_all_creds(user_id)
        return [c for c in credentials if c.provider == provider]

    async def get_authorized_providers(self, user_id: str) -> list[str]:
        credentials = await self.get_all_creds(user_id)
        return list(set(c.provider for c in credentials))

    async def update_creds(self, user_id: str, updated: Credentials) -> None:
        async with await self.locked_user_integrations(user_id):
            current = await self.get_creds_by_id(user_id, updated.id)
            if not current:
                raise ValueError(
                    f"Credentials with ID {updated.id} "
                    f"for user with ID {user_id} not found"
                )
            if type(current) is not type(updated):
                raise TypeError(
                    f"Can not update credentials with ID {updated.id} "
                    f"from type {type(current)} "
                    f"to type {type(updated)}"
                )

            # Ensure no scopes are removed when updating credentials
            if (
                isinstance(updated, OAuth2Credentials)
                and isinstance(current, OAuth2Credentials)
                and not set(updated.scopes).issuperset(current.scopes)
            ):
                raise ValueError(
                    f"Can not update credentials with ID {updated.id} "
                    f"and scopes {current.scopes} "
                    f"to more restrictive set of scopes {updated.scopes}"
                )

            # Update the credentials
            updated_credentials_list = [
                updated if c.id == updated.id else c
                for c in await self.get_all_creds(user_id)
            ]
            await self._set_user_integration_creds(user_id, updated_credentials_list)

    async def delete_creds_by_id(self, user_id: str, credentials_id: str) -> None:
        async with await self.locked_user_integrations(user_id):
            filtered_credentials = [
                c for c in await self.get_all_creds(user_id) if c.id != credentials_id
            ]
            await self._set_user_integration_creds(user_id, filtered_credentials)

    # ============== SYSTEM-MANAGED CREDENTIALS ============== #

    async def get_ayrshare_profile_key(self, user_id: str) -> SecretStr | None:
        """Get the Ayrshare profile key for a user.

        The profile key is used to authenticate API requests to Ayrshare's social media posting service.
        See https://www.ayrshare.com/docs/apis/profiles/overview for more details.

        Args:
            user_id: The ID of the user to get the profile key for

        Returns:
            The profile key as a SecretStr if set, None otherwise
        """
        user_integrations = await self._get_user_integrations(user_id)
        return user_integrations.managed_credentials.ayrshare_profile_key

    async def set_ayrshare_profile_key(self, user_id: str, profile_key: str) -> None:
        """Set the Ayrshare profile key for a user.

        The profile key is used to authenticate API requests to Ayrshare's social media posting service.
        See https://www.ayrshare.com/docs/apis/profiles/overview for more details.

        Args:
            user_id: The ID of the user to set the profile key for
            profile_key: The profile key to set
        """
        _profile_key = SecretStr(profile_key)
        async with self.edit_user_integrations(user_id) as user_integrations:
            user_integrations.managed_credentials.ayrshare_profile_key = _profile_key

    # ===================== OAUTH STATES ===================== #

    async def store_state_token(
        self, user_id: str, provider: str, scopes: list[str], use_pkce: bool = False
    ) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        (code_challenge, code_verifier) = self._generate_code_challenge()

        state = OAuthState(
            token=token,
            provider=provider,
            code_verifier=code_verifier,
            expires_at=int(expires_at.timestamp()),
            scopes=scopes,
        )

        async with self.edit_user_integrations(user_id) as user_integrations:
            user_integrations.oauth_states.append(state)

        async with await self.locked_user_integrations(user_id):

            user_integrations = await self._get_user_integrations(user_id)
            oauth_states = user_integrations.oauth_states
            oauth_states.append(state)
            user_integrations.oauth_states = oauth_states

            await self.db_manager.update_user_integrations(
                user_id=user_id, data=user_integrations
            )

        return token, code_challenge

    def _generate_code_challenge(self) -> tuple[str, str]:
        """
        Generate code challenge using SHA256 from the code verifier.
        Currently only SHA256 is supported.(In future if we want to support more methods we can add them here)
        """
        code_verifier = secrets.token_urlsafe(96)
        sha256_hash = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = base64.urlsafe_b64encode(sha256_hash).decode("utf-8")
        return code_challenge.replace("=", ""), code_verifier

    async def verify_state_token(
        self, user_id: str, token: str, provider: str
    ) -> Optional[OAuthState]:
        async with await self.locked_user_integrations(user_id):
            user_integrations = await self._get_user_integrations(user_id)
            oauth_states = user_integrations.oauth_states

            now = datetime.now(timezone.utc)
            valid_state = next(
                (
                    state
                    for state in oauth_states
                    if secrets.compare_digest(state.token, token)
                    and state.provider == provider
                    and state.expires_at > now.timestamp()
                ),
                None,
            )

            if valid_state:
                # Remove the used state
                oauth_states.remove(valid_state)
                user_integrations.oauth_states = oauth_states
                await self.db_manager.update_user_integrations(
                    user_id, user_integrations
                )
                return valid_state

        return None

    # =================== GET/SET HELPERS =================== #

    @asynccontextmanager
    async def edit_user_integrations(self, user_id: str):
        async with await self.locked_user_integrations(user_id):
            user_integrations = await self._get_user_integrations(user_id)
            yield user_integrations  # yield to allow edits
            await self.db_manager.update_user_integrations(
                user_id=user_id, data=user_integrations
            )

    async def _set_user_integration_creds(
        self, user_id: str, credentials: list[Credentials]
    ) -> None:
        integrations = await self._get_user_integrations(user_id)
        # Remove default credentials from the list
        credentials = [c for c in credentials if c not in DEFAULT_CREDENTIALS]
        integrations.credentials = credentials
        await self.db_manager.update_user_integrations(user_id, integrations)

    async def _get_user_integrations(self, user_id: str) -> UserIntegrations:
        return await self.db_manager.get_user_integrations(user_id=user_id)

    async def locked_user_integrations(self, user_id: str):
        key = (f"user:{user_id}", "integrations")
        locks = await self.locks()
        return locks.locked(key)
