"""
Tests for auto_credentials handling.

These cover ``acquire_auto_credentials`` in ``backend/executor/auto_credentials.py``
— shared by the graph executor and the CoPilot direct-block-execution path.
"""

from unittest.mock import call

import pytest
from pytest_mock import MockerFixture


@pytest.fixture
def google_drive_file_data():
    return {
        "valid": {
            "_credentials_id": "cred-id-123",
            "id": "file-123",
            "name": "test.xlsx",
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
        "chained": {
            "_credentials_id": None,
            "id": "file-456",
            "name": "chained.xlsx",
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
        "missing_key": {
            "id": "file-789",
            "name": "bad.xlsx",
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
    }


@pytest.fixture
def mock_input_model(mocker: MockerFixture):
    """Create a mock input model with get_auto_credentials_fields() returning one field."""
    input_model = mocker.MagicMock()
    input_model.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {
                "provider": "google",
                "type": "oauth2",
                "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
            },
        }
    }
    return input_model


@pytest.fixture
def mock_creds_manager(mocker: MockerFixture):
    manager = mocker.AsyncMock()
    mock_creds = mocker.MagicMock()
    mock_creds.id = "cred-id-123"
    mock_creds.provider = "google"
    mock_creds.type = "oauth2"
    mock_lease = mocker.MagicMock(credentials=mock_creds)
    mock_lease.release = mocker.AsyncMock()
    manager.acquire_lease.return_value = mock_lease
    return manager, mock_creds, mock_lease


@pytest.mark.asyncio
async def test_auto_credentials_happy_path(
    mocker: MockerFixture,
    google_drive_file_data,
    mock_input_model,
    mock_creds_manager,
):
    """When field_data has a valid _credentials_id, credentials should be acquired."""
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, mock_creds, mock_lease = mock_creds_manager
    input_data = {"spreadsheet": google_drive_file_data["valid"]}

    extra_kwargs, leases = await acquire_auto_credentials(
        input_model=mock_input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    manager.acquire_lease.assert_awaited_once_with("user-1", "cred-id-123")
    assert extra_kwargs["credentials"] == mock_creds
    assert mock_lease in leases


@pytest.mark.asyncio
async def test_auto_credentials_field_none_static_raises(
    mocker: MockerFixture,
    mock_input_model,
    mock_creds_manager,
):
    """
    [THE BUG FIX TEST — OPEN-2895]
    When field_data is None and the key IS in input_data (user didn't select a file),
    should raise ValueError instead of silently skipping.
    """
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    # Key is present but value is None = user didn't select a file
    input_data = {"spreadsheet": None}

    with pytest.raises(ValueError, match="No file selected"):
        await acquire_auto_credentials(
            input_model=mock_input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="user-1",
        )


@pytest.mark.asyncio
async def test_auto_credentials_field_absent_skips(
    mocker: MockerFixture,
    mock_input_model,
    mock_creds_manager,
):
    """
    When the field key is NOT in input_data at all (upstream connection),
    should skip without error.
    """
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    # Key not present = connected from upstream block
    input_data = {}

    extra_kwargs, leases = await acquire_auto_credentials(
        input_model=mock_input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    manager.acquire_lease.assert_not_called()
    assert "credentials" not in extra_kwargs
    assert leases == []


@pytest.mark.asyncio
async def test_auto_credentials_chained_cred_id_none(
    mocker: MockerFixture,
    google_drive_file_data,
    mock_input_model,
    mock_creds_manager,
):
    """
    When _credentials_id is explicitly None (chained data from upstream),
    should skip credential acquisition.
    """
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    input_data = {"spreadsheet": google_drive_file_data["chained"]}

    extra_kwargs, leases = await acquire_auto_credentials(
        input_model=mock_input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    manager.acquire_lease.assert_not_called()
    assert "credentials" not in extra_kwargs


@pytest.mark.asyncio
async def test_auto_credentials_missing_cred_id_key_raises(
    mocker: MockerFixture,
    google_drive_file_data,
    mock_input_model,
    mock_creds_manager,
):
    """
    When _credentials_id key is missing entirely from field_data dict,
    should raise ValueError.
    """
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    input_data = {"spreadsheet": google_drive_file_data["missing_key"]}

    with pytest.raises(ValueError, match="Authentication missing"):
        await acquire_auto_credentials(
            input_model=mock_input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="user-1",
        )


@pytest.mark.asyncio
async def test_auto_credentials_ownership_mismatch_error(
    mocker: MockerFixture,
    google_drive_file_data,
    mock_input_model,
    mock_creds_manager,
):
    """
    [SECRT-1772] When acquire() raises ValueError (credential belongs to another user),
    the error message should mention 'not available' (not 'expired').
    """
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    manager.acquire_lease.side_effect = ValueError(
        "Credentials #cred-id-123 for user #user-2 not found"
    )
    input_data = {"spreadsheet": google_drive_file_data["valid"]}

    with pytest.raises(ValueError, match="not available in your account"):
        await acquire_auto_credentials(
            input_model=mock_input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="user-2",
        )


@pytest.mark.asyncio
async def test_auto_credentials_deleted_credential_error(
    mocker: MockerFixture,
    google_drive_file_data,
    mock_input_model,
    mock_creds_manager,
):
    """
    [SECRT-1772] When acquire() raises ValueError (credential was deleted),
    the error message should mention 'not available' (not 'expired').
    """
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    manager.acquire_lease.side_effect = ValueError(
        "Credentials #cred-id-123 for user #user-1 not found"
    )
    input_data = {"spreadsheet": google_drive_file_data["valid"]}

    with pytest.raises(ValueError, match="not available in your account"):
        await acquire_auto_credentials(
            input_model=mock_input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="user-1",
        )


@pytest.mark.asyncio
async def test_auto_credentials_lease_appended(
    mocker: MockerFixture,
    google_drive_file_data,
    mock_input_model,
    mock_creds_manager,
):
    """Lease from acquire_lease() should be returned to guard execution."""
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, _, mock_lease = mock_creds_manager
    input_data = {"spreadsheet": google_drive_file_data["valid"]}

    extra_kwargs, leases = await acquire_auto_credentials(
        input_model=mock_input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    assert len(leases) == 1
    assert leases[0] is mock_lease


@pytest.mark.asyncio
async def test_auto_credentials_multiple_fields(
    mocker: MockerFixture,
    mock_creds_manager,
):
    """When there are multiple auto_credentials fields, only valid ones should acquire."""
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager, mock_creds, mock_lease = mock_creds_manager

    input_model = mocker.MagicMock()
    input_model.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        },
        "credentials2": {
            "field_name": "doc_file",
            "config": {"provider": "google", "type": "oauth2"},
        },
    }

    input_data = {
        "spreadsheet": {
            "_credentials_id": "cred-id-123",
            "id": "file-1",
            "name": "file1.xlsx",
        },
        "doc_file": {
            "_credentials_id": None,
            "id": "file-2",
            "name": "chained.doc",
        },
    }

    extra_kwargs, leases = await acquire_auto_credentials(
        input_model=input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    # Only the first field should have acquired credentials
    manager.acquire_lease.assert_awaited_once_with("user-1", "cred-id-123")
    assert "credentials" in extra_kwargs
    assert "credentials2" not in extra_kwargs


@pytest.mark.asyncio
async def test_acquire_auto_credentials_releases_partial_leases_on_failure(
    mocker: MockerFixture,
):
    """When a later acquisition fails, release every earlier credential lease."""
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager = mocker.AsyncMock()
    good_creds = mocker.MagicMock()
    good_creds.id = "cred-id-good"
    good_creds.provider = "google"
    good_creds.type = "oauth2"
    good_lease = mocker.MagicMock(credentials=good_creds)
    good_lease.release = mocker.AsyncMock()

    async def _acquire(_user_id, cred_id):
        if cred_id == "cred-id-good":
            return good_lease
        raise ValueError(f"bad cred {cred_id}")

    manager.acquire_lease.side_effect = _acquire

    input_model = mocker.MagicMock()
    input_model.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        },
        "credentials2": {
            "field_name": "doc_file",
            "config": {"provider": "google", "type": "oauth2"},
        },
    }

    input_data = {
        "spreadsheet": {
            "_credentials_id": "cred-id-good",
            "id": "file-1",
            "name": "file1.xlsx",
        },
        "doc_file": {
            "_credentials_id": "cred-id-broken",
            "id": "file-2",
            "name": "file2.doc",
        },
    }

    with pytest.raises(ValueError):
        await acquire_auto_credentials(
            input_model=input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="user-1",
        )

    good_lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_auto_credentials_rejects_empty_string_credential_id(
    mocker: MockerFixture,
):
    """Corrupted state: ``_credentials_id`` set to an empty string used to
    slip through ``if cred_id:`` and run without injecting credentials.
    Now it raises so the user re-authenticates rather than executing a
    block that silently has no creds."""
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager = mocker.AsyncMock()

    input_model = mocker.MagicMock()
    input_model.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }

    input_data = {
        "spreadsheet": {
            "_credentials_id": "",  # corrupted empty string
            "id": "file-123",
            "name": "test.xlsx",
        }
    }

    with pytest.raises(ValueError, match="empty or invalid"):
        await acquire_auto_credentials(
            input_model=input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="user-1",
        )

    # Never tried to acquire the (empty) credential.
    manager.acquire_lease.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("1KAv8hhChef7a5ycn6Al1M4DdkiG_PVcKQ_tYkRpGA-I", id="bare-string"),
        pytest.param(42, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(["a", "b"], id="list"),
    ],
)
async def test_acquire_auto_credentials_rejects_non_dict_value_with_type_message(
    mocker: MockerFixture,
    bad_value,
):
    """Cursor Medium (thread PRRT_kwDOJKSTjM58sEDl): the ``else`` branch
    in ``_acquire_auto_credentials`` used to raise "No file selected"
    for ANY truthy non-dict ``field_data`` (e.g. a bare Drive ID
    string).  That message is misleading when the value *was*
    supplied — it's just the wrong shape.  The graph validator catches
    bare strings at save time, but API callers / legacy graphs can
    still reach the runtime.

    Pin the tighter contract: a non-dict value must raise an error
    that names both the field *and* the type it received."""
    from backend.executor.auto_credentials import acquire_auto_credentials

    manager = mocker.AsyncMock()
    input_model = mocker.MagicMock()
    input_model.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }

    input_data = {"spreadsheet": bad_value}

    with pytest.raises(ValueError) as exc_info:
        await acquire_auto_credentials(
            input_model=input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="user-1",
        )

    msg = str(exc_info.value)
    # Must mention the field name.
    assert "spreadsheet" in msg
    # Must describe the actual type rather than the misleading
    # "No file selected" — anchor on the type name so the fix
    # can't silently regress to the old generic message.
    assert type(bad_value).__name__ in msg
    manager.acquire_lease.assert_not_called()


class TestAutoCredentialsOwnerMode:
    """OWNER-mode grant runs resolve the graph's OWN referenced file
    credentials against the graph owner's store, while a file the consumer
    picked themselves still resolves against the consumer."""

    @pytest.mark.asyncio
    async def test_owner_referenced_id_resolves_against_owner(
        self,
        google_drive_file_data,
        mock_input_model,
        mock_creds_manager,
    ):
        from backend.executor.auto_credentials import acquire_auto_credentials

        manager, mock_creds, _ = mock_creds_manager
        consumer_value = {
            **google_drive_file_data["valid"],
            "id": "consumer-resource",
        }
        owner_value = {
            **google_drive_file_data["valid"],
            "id": "owner-resource",
        }
        input_data = {"spreadsheet": consumer_value}

        extra_kwargs, _ = await acquire_auto_credentials(
            input_model=mock_input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="consumer-1",
            credentials_owner_id="owner-1",
            owner_field_values={"spreadsheet": owner_value},
        )

        # cred-id-123 is a graph-referenced id -> resolves against the owner.
        manager.acquire_lease.assert_awaited_once_with("owner-1", "cred-id-123")
        assert extra_kwargs["credentials"] == mock_creds
        assert input_data["spreadsheet"]["id"] == "owner-resource"

    @pytest.mark.asyncio
    async def test_consumer_picked_id_not_in_allowlist_resolves_against_consumer(
        self,
        google_drive_file_data,
        mock_input_model,
        mock_creds_manager,
    ):
        from backend.executor.auto_credentials import acquire_auto_credentials

        manager, _, _ = mock_creds_manager
        input_data = {"spreadsheet": google_drive_file_data["valid"]}

        await acquire_auto_credentials(
            input_model=mock_input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="consumer-1",
            credentials_owner_id="owner-1",
            owner_field_values={"another_picker": google_drive_file_data["valid"]},
        )

        # Not a graph-referenced id -> the consumer's own store, never the owner's.
        manager.acquire_lease.assert_awaited_once_with("consumer-1", "cred-id-123")

    @pytest.mark.asyncio
    async def test_missing_owner_credential_raises_owner_specific_error(
        self,
        google_drive_file_data,
        mock_input_model,
        mock_creds_manager,
    ):
        from backend.executor.auto_credentials import acquire_auto_credentials

        manager, _, _ = mock_creds_manager
        manager.acquire_lease.side_effect = ValueError(
            "Credentials #cred-id-123 for user #owner-1 not found"
        )
        input_data = {"spreadsheet": google_drive_file_data["valid"]}

        with pytest.raises(ValueError, match="run on the graph owner's account"):
            await acquire_auto_credentials(
                input_model=mock_input_model,
                input_data=input_data,
                creds_manager=manager,
                user_id="consumer-1",
                credentials_owner_id="owner-1",
                owner_field_values={"spreadsheet": google_drive_file_data["valid"]},
            )

    @pytest.mark.asyncio
    async def test_same_owner_id_in_another_field_stays_consumer_owned(
        self, mocker, google_drive_file_data, mock_creds_manager
    ):
        from backend.executor.auto_credentials import acquire_auto_credentials

        manager, _, _ = mock_creds_manager
        input_model = mocker.MagicMock()
        input_model.get_auto_credentials_fields.return_value = {
            "first_credentials": {
                "field_name": "first_file",
                "config": {"provider": "google"},
            },
            "second_credentials": {
                "field_name": "second_file",
                "config": {"provider": "google"},
            },
        }
        owner_value = {
            **google_drive_file_data["valid"],
            "id": "owner-resource",
        }
        consumer_value = {
            **google_drive_file_data["valid"],
            "id": "consumer-resource",
        }
        input_data = {
            "first_file": {**consumer_value, "id": "injected-resource"},
            "second_file": consumer_value,
        }

        await acquire_auto_credentials(
            input_model=input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="consumer-1",
            credentials_owner_id="owner-1",
            owner_field_values={"first_file": owner_value},
        )

        assert manager.acquire_lease.await_args_list == [
            call("owner-1", "cred-id-123"),
            call("consumer-1", "cred-id-123"),
        ]
        assert input_data["first_file"]["id"] == "owner-resource"

    @pytest.mark.asyncio
    async def test_provider_mismatch_fails_and_releases_lock(
        self, mock_input_model, mock_creds_manager, google_drive_file_data
    ):
        from backend.executor.auto_credentials import acquire_auto_credentials

        manager, mock_creds, mock_lease = mock_creds_manager
        mock_creds.provider = "dropbox"

        with pytest.raises(ValueError, match="expected 'google'"):
            await acquire_auto_credentials(
                input_model=mock_input_model,
                input_data={"spreadsheet": google_drive_file_data["valid"]},
                creds_manager=manager,
                user_id="consumer-1",
            )

        mock_lease.release.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_type_mismatch_fails_and_releases_lock(
        self, mock_input_model, mock_creds_manager, google_drive_file_data
    ):
        from backend.executor.auto_credentials import acquire_auto_credentials

        manager, mock_creds, mock_lease = mock_creds_manager
        mock_creds.type = "api_key"

        with pytest.raises(ValueError, match="expected 'oauth2'"):
            await acquire_auto_credentials(
                input_model=mock_input_model,
                input_data={"spreadsheet": google_drive_file_data["valid"]},
                creds_manager=manager,
                user_id="consumer-1",
            )

        mock_lease.release.assert_awaited_once()
