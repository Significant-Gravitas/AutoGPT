import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import click
import pytest
from cryptography.fernet import Fernet

from backend.cli.rotate_key import _run_rotation

OLD_KEY = Fernet.generate_key().decode()
NEW_KEY = Fernet.generate_key().decode()


def _encrypted(key: str, data: dict) -> str:
    return Fernet(key.encode()).encrypt(json.dumps(data).encode()).decode()


def _decrypted(key: str, token: str) -> dict:
    return json.loads(Fernet(key.encode()).decrypt(token.encode()))


@pytest.fixture
def rotation(monkeypatch):
    """Drive _run_rotation against in-memory rows, recording every write."""
    import prisma.models

    from backend.data import db as data_db
    from backend.util import settings

    monkeypatch.setattr(data_db, "connect", AsyncMock())
    monkeypatch.setattr(data_db, "disconnect", AsyncMock())

    state = SimpleNamespace(
        new_key=NEW_KEY, users=[], credentials=[], bot_installs=[], writes=[]
    )
    monkeypatch.setattr(
        settings,
        "Settings",
        lambda: SimpleNamespace(secrets=SimpleNamespace(encryption_key=state.new_key)),
    )

    def client(rows: str):
        mock = MagicMock()
        mock.find_many = AsyncMock(side_effect=lambda **_: getattr(state, rows))
        mock.update = AsyncMock(
            side_effect=lambda where, data: state.writes.append((where["id"], data))
        )
        return MagicMock(return_value=mock)

    monkeypatch.setattr(prisma.models.User, "prisma", client("users"))
    monkeypatch.setattr(
        prisma.models.IntegrationCredential, "prisma", client("credentials")
    )
    monkeypatch.setattr(prisma.models.BotInstall, "prisma", client("bot_installs"))
    return state


async def test_every_store_is_readable_under_the_new_key_afterwards(rotation):
    rotation.users = [
        SimpleNamespace(id="u1", integrations=_encrypted(OLD_KEY, {"a": 1}))
    ]
    rotation.credentials = [
        SimpleNamespace(id="c1", encryptedPayload=_encrypted(OLD_KEY, {"b": 2}))
    ]
    rotation.bot_installs = [
        SimpleNamespace(id="b1", credentials=_encrypted(OLD_KEY, {"bot_token": "t"}))
    ]

    await _run_rotation(old_key=OLD_KEY, apply=True)

    written = {id: data for id, data in rotation.writes}
    assert _decrypted(NEW_KEY, written["u1"]["integrations"]) == {"a": 1}
    assert _decrypted(NEW_KEY, written["c1"]["encryptedPayload"]) == {"b": 2}
    assert _decrypted(NEW_KEY, written["b1"]["credentials"]) == {"bot_token": "t"}


async def test_a_dry_run_writes_nothing(rotation, capsys):
    rotation.credentials = [
        SimpleNamespace(id="c1", encryptedPayload=_encrypted(OLD_KEY, {"b": 2}))
    ]

    await _run_rotation(old_key=OLD_KEY, apply=False)

    assert rotation.writes == []
    assert "1 to re-encrypt" in capsys.readouterr().out


async def test_a_dry_run_with_unreadable_values_still_offers_apply(rotation, capsys):
    rotation.credentials = [
        SimpleNamespace(
            id="c1",
            encryptedPayload=_encrypted(Fernet.generate_key().decode(), {"b": 2}),
        ),
        SimpleNamespace(id="c2", encryptedPayload=_encrypted(OLD_KEY, {"c": 3})),
    ]

    await _run_rotation(old_key=OLD_KEY, apply=False)

    out = capsys.readouterr().out
    assert "written under some other key" in out
    assert "Re-run with --apply" in out


async def test_a_value_already_on_the_new_key_is_left_alone(rotation):
    rotation.credentials = [
        SimpleNamespace(id="c1", encryptedPayload=_encrypted(NEW_KEY, {"b": 2}))
    ]

    await _run_rotation(old_key=OLD_KEY, apply=True)

    assert rotation.writes == []


async def test_a_value_neither_key_reads_is_reported_and_left_alone(rotation, capsys):
    other_key = Fernet.generate_key().decode()
    rotation.users = [SimpleNamespace(id="u0", integrations="")]
    rotation.credentials = [
        SimpleNamespace(id="c1", encryptedPayload=_encrypted(other_key, {"b": 2})),
        SimpleNamespace(id="c2", encryptedPayload=_encrypted(OLD_KEY, {"c": 3})),
    ]

    await _run_rotation(old_key=OLD_KEY, apply=True)

    assert [id for id, _ in rotation.writes] == ["c2"]
    out = capsys.readouterr().out
    assert "c1: neither key can read it" in out
    assert (
        "User.integrations: 0 re-encrypted, 0 already on the new key, 0 unreadable"
    ) in out
    assert (
        "IntegrationCredential.encryptedPayload: 1 re-encrypted, "
        "0 already on the new key, 1 unreadable"
    ) in out


@pytest.mark.parametrize(
    "new_key, old_key, message",
    [
        ("", OLD_KEY, "ENCRYPTION_KEY is not set"),
        (OLD_KEY, OLD_KEY, "Change ENCRYPTION_KEY first"),
        (NEW_KEY, "not-a-fernet-key", "Not a valid encryption key"),
    ],
)
async def test_unusable_keys_stop_before_touching_the_database(
    rotation, new_key, old_key, message
):
    rotation.new_key = new_key

    with pytest.raises(click.ClickException, match=message):
        await _run_rotation(old_key=old_key, apply=True)

    assert rotation.writes == []
