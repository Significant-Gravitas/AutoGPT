import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from autogpt_libs.api_key.key_manager import APIKeyManager
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.errors import PrismaError
from prisma.models import APIKey as PrismaAPIKey
from prisma.types import (
    APIKeyCreateInput,
    APIKeyUpdateInput,
    APIKeyWhereInput,
    APIKeyWhereUniqueInput,
)
from pydantic import BaseModel

from backend.data.db import BaseDbModel

logger = logging.getLogger(__name__)


# Some basic exceptions
class APIKeyError(Exception):
    """Base exception for API key operations"""

    pass


class APIKeyNotFoundError(APIKeyError):
    """Raised when an API key is not found"""

    pass


class APIKeyPermissionError(APIKeyError):
    """Raised when there are permission issues with API key operations"""

    pass


class APIKeyValidationError(APIKeyError):
    """Raised when API key validation fails"""

    pass


class APIKey(BaseDbModel):
    name: str
    prefix: str
    key: str
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    permissions: List[APIKeyPermission]
    postfix: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    description: Optional[str] = None
    user_id: str

    @staticmethod
    def from_db(api_key: PrismaAPIKey):
        try:
            return APIKey(
                id=api_key.id,
                name=api_key.name,
                prefix=api_key.prefix,
                postfix=api_key.postfix,
                key=api_key.key,
                status=APIKeyStatus(api_key.status),
                permissions=[APIKeyPermission(p) for p in api_key.permissions],
                created_at=api_key.createdAt,
                last_used_at=api_key.lastUsedAt,
                revoked_at=api_key.revokedAt,
                description=api_key.description,
                user_id=api_key.userId,
            )
        except Exception as e:
            logger.error(f"Error creating APIKey from db: {str(e)}")
            raise APIKeyError(f"Failed to create API key object: {str(e)}")


class APIKeyWithoutHash(BaseModel):
    id: str
    name: str
    prefix: str
    postfix: str
    status: APIKeyStatus
    permissions: List[APIKeyPermission]
    created_at: datetime
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]
    description: Optional[str]
    user_id: str

    @staticmethod
    def from_db(api_key: PrismaAPIKey):
        try:
            return APIKeyWithoutHash(
                id=api_key.id,
                name=api_key.name,
                prefix=api_key.prefix,
                postfix=api_key.postfix,
                status=APIKeyStatus(api_key.status),
                permissions=[APIKeyPermission(p) for p in api_key.permissions],
                created_at=api_key.createdAt,
                last_used_at=api_key.lastUsedAt,
                revoked_at=api_key.revokedAt,
                description=api_key.description,
                user_id=api_key.userId,
            )
        except Exception as e:
            logger.error(f"Error creating APIKeyWithoutHash from db: {str(e)}")
            raise APIKeyError(f"Failed to create API key object: {str(e)}")


async def generate_api_key(
    name: str,
    user_id: str,
    permissions: List[APIKeyPermission],
    description: Optional[str] = None,
) -> tuple[APIKeyWithoutHash, str]:
    """
    Generate a new API key and store it in the database.
    Returns the API key object (without hash) and the plain text key.
    """
    try:
        api_manager = APIKeyManager()
        key = api_manager.generate_api_key()

        api_key = await PrismaAPIKey.prisma().create(
            data=APIKeyCreateInput(
                id=str(uuid.uuid4()),
                name=name,
                prefix=key.prefix,
                postfix=key.postfix,
                key=key.hash,
                permissions=[p for p in permissions],
                description=description,
                userId=user_id,
            )
        )

        api_key_without_hash = APIKeyWithoutHash.from_db(api_key)
        return api_key_without_hash, key.raw
    except PrismaError as e:
        logger.error(f"Database error while generating API key: {str(e)}")
        raise APIKeyError(f"Failed to generate API key: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while generating API key: {str(e)}")
        raise APIKeyError(f"Failed to generate API key: {str(e)}")


async def validate_api_key(plain_text_key: str) -> Optional[APIKey]:
    """
    Validate an API key and return the API key object if valid.
    """
    try:
        if not plain_text_key.startswith(APIKeyManager.PREFIX):
            logger.warning("Invalid API key format")
            return None

        prefix = plain_text_key[: APIKeyManager.PREFIX_LENGTH]
        api_manager = APIKeyManager()

        api_key = await PrismaAPIKey.prisma().find_first(
            where=APIKeyWhereInput(prefix=prefix, status=(APIKeyStatus.ACTIVE))
        )

        if not api_key:
            logger.warning(f"No active API key found with prefix {prefix}")
            return None

        is_valid = api_manager.verify_api_key(plain_text_key, api_key.key)
        if not is_valid:
            logger.warning("API key verification failed")
            return None

        return APIKey.from_db(api_key)
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        raise APIKeyValidationError(f"Failed to validate API key: {str(e)}")


async def revoke_api_key(key_id: str, user_id: str) -> Optional[APIKeyWithoutHash]:
    try:
        api_key = await PrismaAPIKey.prisma().find_unique(where={"id": key_id})

        if not api_key:
            raise APIKeyNotFoundError(f"API key with id {key_id} not found")

        if api_key.userId != user_id:
            raise APIKeyPermissionError(
                "You do not have permission to revoke this API key."
            )

        where_clause: APIKeyWhereUniqueInput = {"id": key_id}
        updated_api_key = await PrismaAPIKey.prisma().update(
            where=where_clause,
            data=APIKeyUpdateInput(
                status=APIKeyStatus.REVOKED, revokedAt=datetime.now(timezone.utc)
            ),
        )

        if updated_api_key:
            return APIKeyWithoutHash.from_db(updated_api_key)
        return None
    except (APIKeyNotFoundError, APIKeyPermissionError) as e:
        raise e
    except PrismaError as e:
        logger.error(f"Database error while revoking API key: {str(e)}")
        raise APIKeyError(f"Failed to revoke API key: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while revoking API key: {str(e)}")
        raise APIKeyError(f"Failed to revoke API key: {str(e)}")


async def list_user_api_keys(user_id: str) -> List[APIKeyWithoutHash]:
    try:
        where_clause: APIKeyWhereInput = {"userId": user_id}

        api_keys = await PrismaAPIKey.prisma().find_many(
            where=where_clause, order={"createdAt": "desc"}
        )

        return [APIKeyWithoutHash.from_db(key) for key in api_keys]
    except PrismaError as e:
        logger.error(f"Database error while listing API keys: {str(e)}")
        raise APIKeyError(f"Failed to list API keys: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while listing API keys: {str(e)}")
        raise APIKeyError(f"Failed to list API keys: {str(e)}")


async def suspend_api_key(key_id: str, user_id: str) -> Optional[APIKeyWithoutHash]:
    try:
        api_key = await PrismaAPIKey.prisma().find_unique(where={"id": key_id})

        if not api_key:
            raise APIKeyNotFoundError(f"API key with id {key_id} not found")

        if api_key.userId != user_id:
            raise APIKeyPermissionError(
                "You do not have permission to suspend this API key."
            )

        where_clause: APIKeyWhereUniqueInput = {"id": key_id}
        updated_api_key = await PrismaAPIKey.prisma().update(
            where=where_clause,
            data=APIKeyUpdateInput(status=APIKeyStatus.SUSPENDED),
        )

        if updated_api_key:
            return APIKeyWithoutHash.from_db(updated_api_key)
        return None
    except (APIKeyNotFoundError, APIKeyPermissionError) as e:
        raise e
    except PrismaError as e:
        logger.error(f"Database error while suspending API key: {str(e)}")
        raise APIKeyError(f"Failed to suspend API key: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while suspending API key: {str(e)}")
        raise APIKeyError(f"Failed to suspend API key: {str(e)}")


def has_permission(api_key: APIKey, required_permission: APIKeyPermission) -> bool:
    try:
        return required_permission in api_key.permissions
    except Exception as e:
        logger.error(f"Error checking API key permissions: {str(e)}")
        return False


async def get_api_key_by_id(key_id: str, user_id: str) -> Optional[APIKeyWithoutHash]:
    try:
        api_key = await PrismaAPIKey.prisma().find_first(
            where=APIKeyWhereInput(id=key_id, userId=user_id)
        )

        if not api_key:
            return None

        return APIKeyWithoutHash.from_db(api_key)
    except PrismaError as e:
        logger.error(f"Database error while getting API key: {str(e)}")
        raise APIKeyError(f"Failed to get API key: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while getting API key: {str(e)}")
        raise APIKeyError(f"Failed to get API key: {str(e)}")


async def update_api_key_permissions(
    key_id: str, user_id: str, permissions: List[APIKeyPermission]
) -> Optional[APIKeyWithoutHash]:
    """
    Update the permissions of an API key.
    """
    try:
        api_key = await PrismaAPIKey.prisma().find_unique(where={"id": key_id})

        if api_key is None:
            raise APIKeyNotFoundError("No such API key found.")

        if api_key.userId != user_id:
            raise APIKeyPermissionError(
                "You do not have permission to update this API key."
            )

        where_clause: APIKeyWhereUniqueInput = {"id": key_id}
        updated_api_key = await PrismaAPIKey.prisma().update(
            where=where_clause,
            data=APIKeyUpdateInput(permissions=permissions),
        )

        if updated_api_key:
            return APIKeyWithoutHash.from_db(updated_api_key)
        return None
    except (APIKeyNotFoundError, APIKeyPermissionError) as e:
        raise e
    except PrismaError as e:
        logger.error(f"Database error while updating API key permissions: {str(e)}")
        raise APIKeyError(f"Failed to update API key permissions: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while updating API key permissions: {str(e)}")
        raise APIKeyError(f"Failed to update API key permissions: {str(e)}")
