import secrets
import string
import hmac

class APIKeyManager:
    def __init__(self, length=32):
        self.length = length
        self.characters = string.ascii_letters + string.digits + "-_"

    def generate_api_key(self):
        """
        Generate a secure API key.
        """
        plain_api_key = ''.join(secrets.choice(self.characters) for _ in range(self.length))
        print("plain_api_key : ", plain_api_key)
        return plain_api_key

    def hash_api_key(self, api_key: str):
        """
          Generate a secure API key using secrets.

          Args:
              api_key (str): The API key to hash

          Returns:
              str: URL-safe encoded API key
        """
        hash_key = secrets.token_urlsafe(32)
        print("hash_key : ", hash_key)
        return secrets.token_urlsafe(32)

    def verify_api_key(self, plain_api_key: str, stored_api_key: str) -> bool:
        """
           Verify if API keys match using secure comparison.

           Args:
               plain_api_key (str): The API key to verify
               stored_api_key (str): The stored API key to compare against

           Returns:
               bool: True if keys match, False otherwise
           """
        print("comapre : ", hmac.compare_digest(plain_api_key, stored_api_key))
        return hmac.compare_digest(plain_api_key, stored_api_key)
