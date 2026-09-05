"""Preview links must not expose credentials or authorize other users."""

from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.fernet import Fernet

from backend.util.desktop_preview import create_preview_link, resolve_preview_link
from backend.util.encryption import JSONCryptor


@pytest.fixture
def cryptor():
    cryptor = JSONCryptor(Fernet.generate_key().decode())
    with patch("backend.util.desktop_preview.JSONCryptor", return_value=cryptor):
        yield cryptor


def test_preview_link_hides_credentials_and_requires_owner(cryptor):
    desktop_url = "https://6080-sandbox.e2b.app/vnc.html?password=private"
    link = create_preview_link("owner", desktop_url)
    assert "password" not in link
    assert "private" not in link
    assert urlsplit(link).path == "/api/proxy/api/desktop-preview"
    token = parse_qs(urlsplit(link).query)["token"][0]
    assert resolve_preview_link("owner", token) == desktop_url
    assert resolve_preview_link("other-user", token) is None
    assert resolve_preview_link("", token) is None
    assert resolve_preview_link("owner", token[:-10]) is None


def test_preview_link_requires_user(cryptor):
    with pytest.raises(ValueError, match="authenticated user"):
        create_preview_link("", "https://preview.example")


@pytest.mark.parametrize(
    "data",
    [
        {"user_id": "owner", "url": "https://preview.example", "purpose": "other"},
        {"user_id": "owner"},
    ],
)
def test_invalid_payload_rejected(cryptor, data):
    assert resolve_preview_link("owner", cryptor.encrypt(data)) is None


def test_expired_preview_rejected(cryptor):
    token = cryptor.fernet.encrypt_at_time(
        b'{"purpose":"e2b-desktop-preview","user_id":"owner","url":"https://preview.example"}',
        current_time=0,
    ).decode()
    assert resolve_preview_link("owner", token) is None
