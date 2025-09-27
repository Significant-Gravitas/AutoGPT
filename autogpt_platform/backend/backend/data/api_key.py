import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from autogpt_libs.api_key.keysmith import APIKeySmith
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey
from prisma.types import APIKeyWhereUniqueInput
from pydantic import BaseModel, Field

from backend.data.includes import MAX_USER_API_KEYS_FETCH
from backend.util.exceptions import NotAuthorizedError, NotFoundError

logger = logging.getLogger(__name__)
keysmith = APIKeySmith()


class APIKeyInfo(BaseModel):
    id: str
    name: str
    head: str = Field(
        description=f"The first {APIKeySmith.HEAD_LENGTH} characters of the key"
    )
    tail: str = Field(
        description=f"The last {APIKeySmith.TAIL_LENGTH} characters of the key"
    )
    status: APIKeyStatus
    permissions: list[APIKeyPermission]
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
            head=api_key.head,
            tail=api_key.tail,
            status=APIKeyStatus(api_key.status),
            permissions=[APIKeyPermission(p) for p in api_key.permissions],
            created_at=api_key.createdAt,
            last_used_at=api_key.lastUsedAt,
            revoked_at=api_key.revokedAt,
            description=api_key.description,
            user_id=api_key.userId,
        )


class APIKeyInfoWithHash(APIKeyInfo):
    hash: str
    salt: str | None = None  # None for legacy keys

    def match(self, plaintext_key: str) -> bool:
        """Returns whether the given key matches this API key object."""
        return keysmith.verify_key(plaintext_key, self.hash, self.salt)

    @staticmethod
    def from_db(api_key: PrismaAPIKey):
        return APIKeyInfoWithHash(
            **APIKeyInfo.from_db(api_key).model_dump(),
            hash=api_key.hash,
            salt=api_key.salt,
        )

    def without_hash(self) -> APIKeyInfo:
        return APIKeyInfo(**self.model_dump(exclude={"hash", "salt"}))


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


async def get_active_api_keys_by_head(head: str) -> list[APIKeyInfoWithHash]:
    results = await PrismaAPIKey.prisma().find_many(
        where={"head": head, "status": APIKeyStatus.ACTIVE}
    )
    return [APIKeyInfoWithHash.from_db(key) for key in results]


async def validate_api_key(plaintext_key: str) -> Optional[APIKeyInfo]:
    """
    Validate an API key and return the API key object if valid and active.
    """
    try:
        if not plaintext_key.startswith(APIKeySmith.PREFIX):
            logger.warning("Invalid API key format")
            return None

        head = plaintext_key[: APIKeySmith.HEAD_LENGTH]
        potential_matches = await get_active_api_keys_by_head(head)

        matched_api_key = next(
            (pm for pm in potential_matches if pm.match(plaintext_key)),
            None,
        )
        if not matched_api_key:
            # API key not found or invalid
            return None

        # Migrate legacy keys to secure format on successful validation
        if matched_api_key.salt is None:
            matched_api_key = await _migrate_key_to_secure_hash(
                plaintext_key, matched_api_key
            )

        return matched_api_key.without_hash()
    except Exception as e:
        logger.error(f"Error while validating API key: {e}")
        raise RuntimeError("Failed to validate API key") from e


async def _migrate_key_to_secure_hash(
    plaintext_key: str, key_obj: APIKeyInfoWithHash
) -> APIKeyInfoWithHash:
    """Replace the SHA256 hash of a legacy API key with a salted Scrypt hash."""
    try:
        new_hash, new_salt = keysmith.hash_key(plaintext_key)
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
    api_key = await PrismaAPIKey.prisma().find_unique(where={"id": key_id})

    if not api_key:
        raise NotFoundError(f"API key with id {key_id} not found")

    if api_key.userId != user_id:
        raise NotAuthorizedError("You do not have permission to revoke this API key.")

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


async def list_user_api_keys(
    user_id: str, limit: int = MAX_USER_API_KEYS_FETCH
) -> list[APIKeyInfo]:
    api_keys = await PrismaAPIKey.prisma().find_many(
        where={"userId": user_id},
        order={"createdAt": "desc"},
        take=limit,
    )

    return [APIKeyInfo.from_db(key) for key in api_keys]


async def suspend_api_key(key_id: str, user_id: str) -> APIKeyInfo:
    selector: APIKeyWhereUniqueInput = {"id": key_id}
    api_key = await PrismaAPIKey.prisma().find_unique(where=selector)

    if not api_key:
        raise NotFoundError(f"API key with id {key_id} not found")

    if api_key.userId != user_id:
        raise NotAuthorizedError("You do not have permission to suspend this API key.")

    updated_api_key = await PrismaAPIKey.prisma().update(
        where=selector, data={"status": APIKeyStatus.SUSPENDED}
    )
    if not updated_api_key:
        raise NotFoundError(f"API key #{key_id} vanished while trying to suspend.")

    return APIKeyInfo.from_db(updated_api_key)


def has_permission(api_key: APIKeyInfo, required_permission: APIKeyPermission) -> bool:
    return required_permission in api_key.permissions


async def get_api_key_by_id(key_id: str, user_id: str) -> Optional[APIKeyInfo]:
    api_key = await PrismaAPIKey.prisma().find_first(
        where={"id": key_id, "userId": user_id}
    )

    if not api_key:
        return None

    return APIKeyInfo.from_db(api_key)


async def update_api_key_permissions(
    key_id: str, user_id: str, permissions: list[APIKeyPermission]
) -> APIKeyInfo:
    """
    Update the permissions of an API key.
    """
    api_key = await PrismaAPIKey.prisma().find_unique(where={"id": key_id})

    if api_key is None:
        raise NotFoundError("No such API key found.")

    if api_key.userId != user_id:
        raise NotAuthorizedError("You do not have permission to update this API key.")

    updated_api_key = await PrismaAPIKey.prisma().update(
        where={"id": key_id},
        data={"permissions": permissions},
    )
    if not updated_api_key:
        raise NotFoundError(f"API key #{key_id} vanished while trying to update.")

    return APIKeyInfo.from_db(updated_api_key)
