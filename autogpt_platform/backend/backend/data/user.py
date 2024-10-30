import json
from cryptography.fernet import Fernet

from typing import Optional

from autogpt_libs.supabase_integration_credentials_store.types import UserMetadataRaw
from backend.util.settings import Settings
from fastapi import HTTPException
from prisma import Json
from prisma.models import User

from backend.data.db import prisma

DEFAULT_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
DEFAULT_EMAIL = "default@example.com"


async def get_or_create_user(user_data: dict) -> User:
    user_id = user_data.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")

    user_email = user_data.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="Email not found in token")

    user = await prisma.user.find_unique(where={"id": user_id})
    if not user:
        user = await prisma.user.create(
            data={
                "id": user_id,
                "email": user_email,
                "name": user_data.get("user_metadata", {}).get("name"),
            }
        )
    return User.model_validate(user)


async def get_user_by_id(user_id: str) -> Optional[User]:
    user = await prisma.user.find_unique(where={"id": user_id})
    return User.model_validate(user) if user else None


async def create_default_user() -> Optional[User]:
    user = await prisma.user.find_unique(where={"id": DEFAULT_USER_ID})
    if not user:
        user = await prisma.user.create(
            data={
                "id": DEFAULT_USER_ID,
                "email": "default@example.com",
                "name": "Default User",
            }
        )
    return User.model_validate(user)


ENCRYPTION_KEY = Settings().secrets.encryption_key

class EncryptedJson:
    def __init__(self, key: Optional[str] = None):
        # Use provided key or get from environment
        self.key = key or ENCRYPTION_KEY
        if not self.key:
            raise ValueError(
                "Encryption key must be provided or set in ENCRYPTION_KEY environment variable"
            )
        self.fernet = Fernet(
            self.key.encode() if isinstance(self.key, str) else self.key
        )

    def encrypt(self, data: dict) -> str:
        """Encrypt dictionary data to string"""
        json_str = json.dumps(data)
        encrypted = self.fernet.encrypt(json_str.encode())
        return encrypted.decode()

    def decrypt(self, encrypted_str: str) -> dict:
        """Decrypt string to dictionary"""
        if not encrypted_str:
            return {}
        decrypted = self.fernet.decrypt(encrypted_str.encode())
        return json.loads(decrypted.decode())


# Initialize encryption
json_encryptor = EncryptedJson()


def encrypt_metadata(metadata: UserMetadataRaw) -> str:
    """Encrypt metadata Pydantic model to string"""
    return json_encryptor.encrypt(metadata.model_dump())


def decrypt_metadata(metadata: str) -> UserMetadataRaw:
    """Decrypt string to metadata Pydantic model"""
    decrypted_dict = json_encryptor.decrypt(metadata)
    return UserMetadataRaw.model_validate(decrypted_dict)


async def get_user_metadata(user_id: str) -> UserMetadataRaw:
    user = await User.prisma().find_unique_or_raise(
        where={"id": user_id},
    )

    # Decrypt the metadata
    metadata = user.metadata

    if not metadata:
        return UserMetadataRaw()
    else:
        decrypted_metadata = decrypt_metadata(metadata)
        return decrypted_metadata

async def update_user_metadata(user_id: str, metadata: UserMetadataRaw):
    # Encrypt the metadata
    encrypted_metadata = encrypt_metadata(metadata)

    await User.prisma().update(
        where={"id": user_id},
        data={"metadata": encrypted_metadata},
    )
