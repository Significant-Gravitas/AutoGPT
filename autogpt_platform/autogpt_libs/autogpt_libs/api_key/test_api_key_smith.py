import hashlib

from autogpt_libs.api_key.api_key_smith import APIKeySmith


def test_generate_api_key():
    api_key_smith = APIKeySmith()
    key = api_key_smith.generate_key()

    assert key.key.startswith(api_key_smith.PREFIX)
    assert key.head == key.key[: api_key_smith.HEAD_LENGTH]
    assert key.tail == key.key[-api_key_smith.TAIL_LENGTH :]
    assert len(key.hash) == 64  # 32 bytes hex encoded
    assert len(key.salt) == 64  # 32 bytes hex encoded


def test_verify_new_secure_key():
    api_key_smith = APIKeySmith()
    key = api_key_smith.generate_key()

    # Test correct key validates
    assert api_key_smith.verify_key(key.key, key.hash, key.salt) is True

    # Test wrong key fails
    wrong_key = f"{api_key_smith.PREFIX}wrongkey123"
    assert api_key_smith.verify_key(wrong_key, key.hash, key.salt) is False


def test_verify_legacy_key():
    api_key_smith = APIKeySmith()
    legacy_key = f"{api_key_smith.PREFIX}legacykey123"
    legacy_hash = hashlib.sha256(legacy_key.encode()).hexdigest()

    # Test legacy key validates without salt
    assert api_key_smith.verify_key(legacy_key, legacy_hash) is True

    # Test wrong legacy key fails
    wrong_key = f"{api_key_smith.PREFIX}wronglegacy"
    assert api_key_smith.verify_key(wrong_key, legacy_hash) is False


def test_rehash_existing_key():
    api_key_smith = APIKeySmith()
    legacy_key = f"{api_key_smith.PREFIX}migratekey123"

    # Migrate the legacy key
    new_hash, new_salt = api_key_smith.hash_key(legacy_key)

    # Verify migrated key works
    assert api_key_smith.verify_key(legacy_key, new_hash, new_salt) is True

    # Verify different key fails with migrated hash
    wrong_key = f"{api_key_smith.PREFIX}wrongkey"
    assert api_key_smith.verify_key(wrong_key, new_hash, new_salt) is False


def test_invalid_key_prefix():
    api_key_smith = APIKeySmith()
    key = api_key_smith.generate_key()

    # Test key without proper prefix fails
    invalid_key = "invalid_prefix_key"
    assert api_key_smith.verify_key(invalid_key, key.hash, key.salt) is False


def test_secure_hash_requires_salt():
    api_key_smith = APIKeySmith()
    key = api_key_smith.generate_key()

    # Secure hash without salt should fail
    assert api_key_smith.verify_key(key.key, key.hash) is False


def test_invalid_salt_format():
    api_key_smith = APIKeySmith()
    key = api_key_smith.generate_key()

    # Invalid salt format should fail gracefully
    assert api_key_smith.verify_key(key.key, key.hash, "invalid_hex") is False
