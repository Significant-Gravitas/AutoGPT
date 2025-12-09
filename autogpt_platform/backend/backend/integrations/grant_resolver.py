"""
Grant-Based Credential Resolver.

Resolves credentials during agent execution based on credential grants.
External applications can only use credentials they have been granted access to,
and only for the scopes that were granted.

Credentials are NEVER exposed to external applications - this resolver
provides the credentials to the execution engine internally.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from prisma.enums import CredentialGrantPermission
from prisma.models import CredentialGrant

from backend.data import credential_grants as grants_db
from backend.data.db import prisma
from backend.data.model import Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager

logger = logging.getLogger(__name__)


class GrantValidationError(Exception):
    """Raised when a grant is invalid or lacks required permissions."""

    pass


class CredentialNotFoundError(Exception):
    """Raised when a credential referenced by a grant is not found."""

    pass


class ScopeMismatchError(Exception):
    """Raised when the grant doesn't cover required scopes."""

    pass


class GrantBasedCredentialResolver:
    """
    Resolves credentials for agent execution based on credential grants.

    This resolver validates that:
    1. The grant exists and is valid (not revoked/expired)
    2. The grant has USE permission
    3. The grant covers the required scopes (if specified)
    4. The underlying credential exists

    Then it provides the credential to the execution engine internally.
    The credential value is NEVER exposed to external applications.
    """

    def __init__(
        self,
        user_id: str,
        client_id: str,
        grant_ids: list[str],
    ):
        """
        Initialize the resolver.

        Args:
            user_id: User ID who owns the credentials
            client_id: Database ID of the OAuth client
            grant_ids: List of grant IDs the client is using for this execution
        """
        self.user_id = user_id
        self.client_id = client_id
        self.grant_ids = grant_ids
        self._grants: dict[str, CredentialGrant] = {}
        self._credentials_manager = IntegrationCredentialsManager()
        self._initialized = False

    async def initialize(self) -> None:
        """
        Load and validate all grants.

        This should be called before any credential resolution.

        Raises:
            GrantValidationError: If any grant is invalid
        """
        now = datetime.now(timezone.utc)

        for grant_id in self.grant_ids:
            grant = await grants_db.get_credential_grant(
                grant_id=grant_id,
                user_id=self.user_id,
                client_id=self.client_id,
            )

            if not grant:
                raise GrantValidationError(f"Grant {grant_id} not found")

            # Check if revoked
            if grant.revokedAt:
                raise GrantValidationError(f"Grant {grant_id} has been revoked")

            # Check if expired
            if grant.expiresAt and grant.expiresAt < now:
                raise GrantValidationError(f"Grant {grant_id} has expired")

            # Check USE permission
            if CredentialGrantPermission.USE not in grant.permissions:
                raise GrantValidationError(
                    f"Grant {grant_id} does not have USE permission"
                )

            self._grants[grant_id] = grant

        self._initialized = True
        logger.info(
            f"Initialized grant resolver with {len(self._grants)} grants "
            f"for user {self.user_id}, client {self.client_id}"
        )

    async def resolve_credential(
        self,
        credential_id: str,
        required_scopes: Optional[list[str]] = None,
    ) -> Credentials:
        """
        Resolve a credential for agent execution.

        This method:
        1. Finds a grant that covers this credential
        2. Validates the grant covers required scopes
        3. Retrieves the actual credential
        4. Updates grant usage tracking

        Args:
            credential_id: ID of the credential to resolve
            required_scopes: Optional list of scopes the credential must have

        Returns:
            The resolved Credentials object

        Raises:
            GrantValidationError: If no valid grant covers this credential
            ScopeMismatchError: If the grant doesn't cover required scopes
            CredentialNotFoundError: If the underlying credential doesn't exist
        """
        if not self._initialized:
            raise RuntimeError("Resolver not initialized. Call initialize() first.")

        # Find a grant that covers this credential
        matching_grant: Optional[CredentialGrant] = None
        for grant in self._grants.values():
            if grant.credentialId == credential_id:
                matching_grant = grant
                break

        if not matching_grant:
            raise GrantValidationError(f"No grant found for credential {credential_id}")

        # Validate scopes if required
        if required_scopes:
            granted_scopes = set(matching_grant.grantedScopes)
            required_scopes_set = set(required_scopes)

            missing_scopes = required_scopes_set - granted_scopes
            if missing_scopes:
                raise ScopeMismatchError(
                    f"Grant {matching_grant.id} is missing required scopes: "
                    f"{', '.join(missing_scopes)}"
                )

        # Get the actual credential
        credentials = await self._credentials_manager.get(
            user_id=self.user_id,
            credentials_id=credential_id,
            lock=True,
        )

        if not credentials:
            raise CredentialNotFoundError(
                f"Credential {credential_id} not found for user {self.user_id}"
            )

        # Update last used timestamp for the grant
        await grants_db.update_grant_last_used(matching_grant.id)

        logger.debug(
            f"Resolved credential {credential_id} via grant {matching_grant.id} "
            f"for client {self.client_id}"
        )

        return credentials

    async def get_available_credentials(self) -> list[dict]:
        """
        Get list of available credentials based on grants.

        Returns a list of credential metadata (NOT the actual credential values).

        Returns:
            List of dicts with credential metadata
        """
        if not self._initialized:
            raise RuntimeError("Resolver not initialized. Call initialize() first.")

        credentials_info = []
        for grant in self._grants.values():
            credentials_info.append(
                {
                    "grant_id": grant.id,
                    "credential_id": grant.credentialId,
                    "provider": grant.provider,
                    "granted_scopes": grant.grantedScopes,
                }
            )

        return credentials_info

    def get_grant_for_credential(self, credential_id: str) -> Optional[CredentialGrant]:
        """
        Get the grant for a specific credential.

        Args:
            credential_id: ID of the credential

        Returns:
            CredentialGrant or None if not found
        """
        for grant in self._grants.values():
            if grant.credentialId == credential_id:
                return grant
        return None


async def create_resolver_from_oauth_token(
    user_id: str,
    client_public_id: str,
    grant_ids: Optional[list[str]] = None,
) -> GrantBasedCredentialResolver:
    """
    Create a credential resolver from OAuth token context.

    This is a convenience function for creating a resolver from
    the context available in OAuth-authenticated requests.

    Args:
        user_id: User ID from the OAuth token
        client_public_id: Public client ID from the OAuth token
        grant_ids: Optional list of grant IDs to use

    Returns:
        Initialized GrantBasedCredentialResolver
    """
    # Look up the OAuth client database ID from the public client ID
    client = await prisma.oauthclient.find_unique(where={"clientId": client_public_id})
    if not client:
        raise GrantValidationError(f"OAuth client {client_public_id} not found")

    # If no grant IDs specified, get all grants for this client+user
    if grant_ids is None:
        grants = await grants_db.get_grants_for_user_client(
            user_id=user_id,
            client_id=client.id,
            include_revoked=False,
            include_expired=False,
        )
        grant_ids = [g.id for g in grants]

    resolver = GrantBasedCredentialResolver(
        user_id=user_id,
        client_id=client.id,
        grant_ids=grant_ids,
    )
    await resolver.initialize()

    return resolver
