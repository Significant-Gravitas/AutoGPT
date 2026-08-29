import hashlib

from autogpt_libs.api_key.keysmith import APIKeySmith


def test_generate_api_key():
    keysmith = APIKeySmith()
    key = keysmith.generate_key()

    assert key.key.startswith(keysmith.PREFIX)
    assert key.head == key.key[: keysmith.HEAD_LENGTH]
    assert key.tail == key.key[-keysmith.TAIL_LENGTH :]
    assert len(key.hash) == 64  # 32 bytes hex encoded
    assert len(key.salt) == 64  # 32 bytes hex encoded


def test_verify_new_secure_key():
    keysmith = APIKeySmith()
    key = keysmith.generate_key()

    # Test correct key validates
    assert keysmith.verify_key(key.key, key.hash, key.salt) is True

    # Test wrong key fails
    wrong_key = f"{keysmith.PREFIX}wrongkey123"
    assert keysmith.verify_key(wrong_key, key.hash, key.salt) is False


def test_verify_legacy_key():
    keysmith = APIKeySmith()
    legacy_key = f"{keysmith.PREFIX}legacykey123"
    legacy_hash = hashlib.sha256(legacy_key.encode()).hexdigest()

    # Test legacy key validates without salt
    assert keysmith.verify_key(legacy_key, legacy_hash) is True

    # Test wrong legacy key fails
    wrong_key = f"{keysmith.PREFIX}wronglegacy"
    assert keysmith.verify_key(wrong_key, legacy_hash) is False


def test_rehash_existing_key():
    keysmith = APIKeySmith()
    legacy_key = f"{keysmith.PREFIX}migratekey123"

    # Migrate the legacy key
    new_hash, new_salt = keysmith.hash_key(legacy_key)

    # Verify migrated key works
    assert keysmith.verify_key(legacy_key, new_hash, new_salt) is True

    # Verify different key fails with migrated hash
    wrong_key = f"{keysmith.PREFIX}wrongkey"
    assert keysmith.verify_key(wrong_key, new_hash, new_salt) is False


def test_invalid_key_prefix():
    keysmith = APIKeySmith()
    key = keysmith.generate_key()

    # Test key without proper prefix fails
    invalid_key = "invalid_prefix_key"
    assert keysmith.verify_key(invalid_key, key.hash, key.salt) is False


def test_secure_hash_requires_salt():
    keysmith = APIKeySmith()
    key = keysmith.generate_key()

    # Secure hash without salt should fail
    assert keysmith.verify_key(key.key, key.hash) is False


def test_invalid_salt_format():
    keysmith = APIKeySmith()
    key = keysmith.generate_key()

    # Invalid salt format should fail gracefully
    assert keysmith.verify_key(key.key, key.hash, "invalid_hex") is False
