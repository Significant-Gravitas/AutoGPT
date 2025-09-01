import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from autogpt_libs.api_key.api_key_smith import APIKeySmith
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.errors import PrismaError
from prisma.models import APIKey as PrismaAPIKey
from prisma.types import APIKeyWhereUniqueInput

from backend.data.db import BaseDbModel
from backend.util.exceptions import NotAuthorizedError, NotFoundError

logger = logging.getLogger(__name__)
keysmith = APIKeySmith()


# Some basic exceptions
class APIKeyError(Exception):
    """Base exception for API key operations"""

    pass


class APIKeyInfo(BaseDbModel):
    name: str
    prefix: str
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    permissions: list[APIKeyPermission]
    postfix: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    description: Optional[str] = None
    user_id: str

    @staticmethod
    def from_db(api_key: PrismaAPIKey):
        return APIKeyInfo(
            id=api_key.id,
            name=api_key.name,
            prefix=api_key.head,
            postfix=api_key.tail,
            status=APIKeyStatus(api_key.status),
            permissions=[APIKeyPermission(p) for p in api_key.permissions],
            created_at=api_key.createdAt,
            last_used_at=api_key.lastUsedAt,
            revoked_at=api_key.revokedAt,
            description=api_key.description,
            user_id=api_key.userId,
        )


class APIKeyWithHash(APIKeyInfo):
    hash: str

    @staticmethod
    def from_db(api_key: PrismaAPIKey):
        return APIKeyWithHash(
            **APIKeyInfo.from_db(api_key).model_dump(),
            hash=api_key.hash,
        )


async def create_api_key(
    name: str,
    user_id: str,
    permissions: list[APIKeyPermission],
    description: Optional[str] = None,
) -> tuple[APIKeyInfo, str]:
    """
    Generate a new API key and store it in the database.
    Returns the API key object (without hash) and the plain text key.
    """
    try:
        generated_key = keysmith.generate_key()

        saved_key_obj = await PrismaAPIKey.prisma().create(
            data={
                "id": str(uuid.uuid4()),
                "name": name,
                "head": generated_key.head,
                "tail": generated_key.tail,
                "hash": generated_key.hash,
                "salt": generated_key.salt,
                "permissions": [p for p in permissions],
                "description": description,
                "userId": user_id,
            }
        )

        return APIKeyInfo.from_db(saved_key_obj), generated_key.key
    except PrismaError as e:
        logger.error(f"Database error while generating API key: {str(e)}")
        raise APIKeyError(f"Failed to generate API key: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while generating API key: {str(e)}")
        raise APIKeyError(f"Failed to generate API key: {str(e)}")


async def validate_api_key(raw_key: str) -> Optional[APIKeyInfo]:
    """
    Validate an API key and return the API key object if valid.
    """
    try:
        if not raw_key.startswith(APIKeySmith.PREFIX):
            logger.warning("Invalid API key format")
            return None

        head = raw_key[: APIKeySmith.HEAD_LENGTH]

        possible_matches = await PrismaAPIKey.prisma().find_many(
            where={"head": head, "status": APIKeyStatus.ACTIVE}
        )

        found_api_key = next(
            (
                known_key
                for known_key in possible_matches
                if keysmith.verify_key(raw_key, known_key.hash, known_key.salt)
            ),
            None,
        )
        if not found_api_key:
            # API key not found or invalid
            return None

        # Migrate legacy keys to secure format on successful validation
        if found_api_key.salt is None:
            found_api_key = await _migrate_key_to_secure_hash(raw_key, found_api_key)

        return APIKeyInfo.from_db(found_api_key)
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        raise RuntimeError("Failed to validate API key") from e


async def _migrate_key_to_secure_hash(
    raw_key: str, key_obj: PrismaAPIKey
) -> PrismaAPIKey:
    """Replace the SHA256 hash of a legacy API key with a salted Scrypt hash."""
    try:
        new_hash, new_salt = keysmith.hash_key(raw_key)
        await PrismaAPIKey.prisma().update(
            where={"id": key_obj.id}, data={"hash": new_hash, "salt": new_salt}
        )
        logger.info(f"Migrated legacy API key #{key_obj.id} to secure format")
        # Update the API key object with new values for return
        key_obj.hash = new_hash
        key_obj.salt = new_salt
    except Exception as e:
        logger.error(f"Failed to migrate legacy API key #{key_obj.id}: {e}")

    return key_obj


async def revoke_api_key(key_id: str, user_id: str) -> APIKeyInfo:
    try:
        api_key = await PrismaAPIKey.prisma().find_unique(where={"id": key_id})

        if not api_key:
            raise NotFoundError(f"API key with id {key_id} not found")

        if api_key.userId != user_id:
            raise NotAuthorizedError(
                "You do not have permission to revoke this API key."
            )

        updated_api_key = await PrismaAPIKey.prisma().update(
            where={"id": key_id},
            data={
                "status": APIKeyStatus.REVOKED,
                "revokedAt": datetime.now(timezone.utc),
            },
        )
        if not updated_api_key:
            raise NotFoundError(f"API key #{key_id} vanished while trying to revoke.")

        return APIKeyInfo.from_db(updated_api_key)
    except (NotFoundError, NotAuthorizedError) as e:
        raise e
    except PrismaError as e:
        logger.error(f"Database error while revoking API key: {str(e)}")
        raise APIKeyError(f"Failed to revoke API key: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while revoking API key: {str(e)}")
        raise APIKeyError(f"Failed to revoke API key: {str(e)}")


async def list_user_api_keys(user_id: str) -> list[APIKeyInfo]:
    try:
        api_keys = await PrismaAPIKey.prisma().find_many(
            where={"userId": user_id}, order={"createdAt": "desc"}
        )

        return [APIKeyInfo.from_db(key) for key in api_keys]
    except PrismaError as e:
        logger.error(f"Database error while listing API keys: {str(e)}")
        raise APIKeyError(f"Failed to list API keys: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while listing API keys: {str(e)}")
        raise APIKeyError(f"Failed to list API keys: {str(e)}")


async def suspend_api_key(key_id: str, user_id: str) -> APIKeyInfo:
    selector: APIKeyWhereUniqueInput = {"id": key_id}
    try:
        api_key = await PrismaAPIKey.prisma().find_unique(where=selector)

        if not api_key:
            raise NotFoundError(f"API key with id {key_id} not found")

        if api_key.userId != user_id:
            raise NotAuthorizedError(
                "You do not have permission to suspend this API key."
            )

        updated_api_key = await PrismaAPIKey.prisma().update(
            where=selector, data={"status": APIKeyStatus.SUSPENDED}
        )
        if not updated_api_key:
            raise NotFoundError(f"API key #{key_id} vanished while trying to suspend.")

        return APIKeyInfo.from_db(updated_api_key)
    except (NotFoundError, NotAuthorizedError) as e:
        raise e
    except PrismaError as e:
        logger.error(f"Database error while suspending API key: {str(e)}")
        raise APIKeyError(f"Failed to suspend API key: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while suspending API key: {str(e)}")
        raise APIKeyError(f"Failed to suspend API key: {str(e)}")


def has_permission(api_key: APIKeyInfo, required_permission: APIKeyPermission) -> bool:
    try:
        return required_permission in api_key.permissions
    except Exception as e:
        logger.error(f"Error checking API key permissions: {str(e)}")
        return False


async def get_api_key_by_id(key_id: str, user_id: str) -> Optional[APIKeyInfo]:
    try:
        api_key = await PrismaAPIKey.prisma().find_first(
            where={"id": key_id, "userId": user_id}
        )

        if not api_key:
            return None

        return APIKeyInfo.from_db(api_key)
    except PrismaError as e:
        logger.error(f"Database error while getting API key: {str(e)}")
        raise APIKeyError(f"Failed to get API key: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while getting API key: {str(e)}")
        raise APIKeyError(f"Failed to get API key: {str(e)}")


async def update_api_key_permissions(
    key_id: str, user_id: str, permissions: list[APIKeyPermission]
) -> APIKeyInfo:
    """
    Update the permissions of an API key.
    """
    try:
        api_key = await PrismaAPIKey.prisma().find_unique(where={"id": key_id})

        if api_key is None:
            raise NotFoundError("No such API key found.")

        if api_key.userId != user_id:
            raise NotAuthorizedError(
                "You do not have permission to update this API key."
            )

        updated_api_key = await PrismaAPIKey.prisma().update(
            where={"id": key_id},
            data={"permissions": permissions},
        )
        if not updated_api_key:
            raise NotFoundError(f"API key #{key_id} vanished while trying to update.")

        return APIKeyInfo.from_db(updated_api_key)
    except (NotFoundError, NotAuthorizedError) as e:
        raise e
    except PrismaError as e:
        logger.error(f"Database error while updating API key permissions: {str(e)}")
        raise APIKeyError(f"Failed to update API key permissions: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while updating API key permissions: {str(e)}")
        raise APIKeyError(f"Failed to update API key permissions: {str(e)}")
