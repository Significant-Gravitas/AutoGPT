import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Union
from enum import Enum
from pydantic import BaseModel

from prisma.models import APIKey as PrismaAPIKey
from backend.data.db import BaseDbModel, transaction
from autogpt_libs.api_key.key_manager import APIKeyManager

logger = logging.getLogger(__name__)

class APIKeyPermission(str, Enum):
    EXECUTE_GRAPH = "EXECUTE_GRAPH"
    READ_GRAPH = "READ_GRAPH"
    EXECUTE_BLOCK = "EXECUTE_BLOCK"
    READ_BLOCK = "READ_BLOCK"

class APIKeyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    SUSPENDED = "SUSPENDED"

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
            user_id=api_key.userId
        )

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
            user_id=api_key.userId
        )

# --------------------- Model functions --------------------- #

async def generate_api_key(
    name: str,
    user_id: str,
    permissions: List[APIKeyPermission],
    description: Optional[str] = None
) -> tuple[APIKeyWithoutHash, str]:
    """
    Generate a new API key and store it in the database.
    Returns the API key object (without hash) and the plain text key.
    """
    api_manager = APIKeyManager()
    plain_text_key = api_manager.generate_api_key()
    prefix = plain_text_key[:8]
    postfix = plain_text_key[-8:]
    hashed_key = api_manager.hash_api_key(plain_text_key)

    api_key = await PrismaAPIKey.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "name": name,
            "prefix": prefix,
            "postfix": postfix,
            "key": hashed_key,
            "permissions": [p.value for p in permissions],
            "description": description,
            "userId": user_id
        }
    )

    api_key_without_hash = APIKeyWithoutHash.from_db(api_key)

    return api_key_without_hash, plain_text_key

async def validate_api_key(plain_text_key: str) -> Optional[APIKey]:
    """
    Validate an API key and return the API key object if valid.
    """
    prefix = plain_text_key[:8]
    api_manager = APIKeyManager()

    api_key = await PrismaAPIKey.prisma().find_first(
        where={
            "prefix": prefix,
            "status": APIKeyStatus.ACTIVE.value
        }
    )

    if not api_key:
        return None

    is_valid = api_manager.verify_api_key(plain_text_key,api_key.key)
    if not is_valid:
        return None

    return APIKey.from_db(api_key)

async def revoke_api_key(key_id: str, user_id: str) -> APIKeyWithoutHash:
    api_key = await PrismaAPIKey.prisma().update(
        where={
            "id": key_id,
            "userId": user_id
        },
        data={
            "status": APIKeyStatus.REVOKED.value,
            "revokedAt": datetime.now(timezone.utc)
        }
    )

    return APIKeyWithoutHash.from_db(api_key)

async def list_user_api_keys(user_id: str) -> List[APIKeyWithoutHash]:
    api_keys = await PrismaAPIKey.prisma().find_many(
        where={"userId": user_id},
        order={"createdAt": "desc"}
    )

    return [APIKeyWithoutHash.from_db(key) for key in api_keys]

async def suspend_api_key(key_id: str, user_id: str) -> APIKeyWithoutHash:
    api_key = await PrismaAPIKey.prisma().update(
        where={
            "id": key_id,
            "userId": user_id
        },
        data={"status": APIKeyStatus.SUSPENDED.value}
    )

    return APIKeyWithoutHash.from_db(api_key)

def has_permission(api_key: APIKey, required_permission: APIKeyPermission) -> bool:
    return required_permission in api_key.permissions

async def get_api_key_by_id(key_id: str, user_id: str) -> Optional[APIKeyWithoutHash]:
    api_key = await PrismaAPIKey.prisma().find_first(
        where={
            "id": key_id,
            "userId": user_id
        }
    )

    return APIKeyWithoutHash.from_db(api_key) if api_key else None

async def update_api_key_permissions(
    key_id: str,
    user_id: str,
    permissions: List[APIKeyPermission]
) -> APIKeyWithoutHash:
    """
    Update the permissions of an API key.
    """
    api_key = await PrismaAPIKey.prisma().update(
        where={
            "id": key_id,
            "userId": user_id
        },
        data={"permissions": [p.value for p in permissions]}
    )

    return APIKeyWithoutHash.from_db(api_key)
