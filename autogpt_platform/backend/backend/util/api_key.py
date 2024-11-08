import secrets
import string
import bcrypt

class APIKeyManager:
    def __init__(self, length=32):
        self.length = length
        self.characters = string.ascii_letters + string.digits + "-_"

    def generate_api_key(self):
        """
        Generate a secure API key.
        """
        plain_api_key = ''.join(secrets.choice(self.characters) for _ in range(self.length))
        return plain_api_key

    def hash_api_key(self, api_key: str) -> bytes:
        """
        Hash an API key using bcrypt.

        Args:
            api_key (str): The API key to hash

        Returns:
            bytes: The hashed API key
        """
        api_key_bytes = api_key.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed_key = bcrypt.hashpw(api_key_bytes, salt)
        return hashed_key.decode() #decoding the encoding

    def verify_api_key(self, plain_api_key: str, hashed_api_key: bytes) -> bool:
        """
        Verify if a plain API key matches its hashed version.

        Args:
            plain_api_key (str): The plain API key to verify
            hashed_api_key (bytes): The hashed API key to compare against

        Returns:
            bool: True if the API key matches, False otherwise
        """
        return bcrypt.checkpw(plain_api_key.encode('utf-8'), hashed_api_key)
