
import hashlib


def calculate_sha256(stream):
    sha256_hash = hashlib.sha256()
    for chunk in stream:
        sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def is_valid_int(value: str) -> bool:
    """Check if the value is a valid integer

    Args:
        value (str): The value to check

    Returns:
        bool: True if the value is a valid integer, False otherwise
    """
    try:
        int(value)
        return True
    except ValueError:
        return False
