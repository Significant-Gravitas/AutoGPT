"""
Tests for auto_credentials handling in execute_node().

These test the _acquire_auto_credentials() helper function extracted from
execute_node() (manager.py lines 273-308).
"""

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
    mock_lock = mocker.AsyncMock()
    mock_creds = mocker.MagicMock()
    mock_creds.id = "cred-id-123"
    mock_creds.provider = "google"
    manager.acquire.return_value = (mock_creds, mock_lock)
    return manager, mock_creds, mock_lock


@pytest.mark.asyncio
async def test_auto_credentials_happy_path(
    mocker: MockerFixture,
    google_drive_file_data,
    mock_input_model,
    mock_creds_manager,
):
    """When field_data has a valid _credentials_id, credentials should be acquired."""
    from backend.executor.manager import _acquire_auto_credentials

    manager, mock_creds, mock_lock = mock_creds_manager
    input_data = {"spreadsheet": google_drive_file_data["valid"]}

    extra_kwargs, locks = await _acquire_auto_credentials(
        input_model=mock_input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    manager.acquire.assert_called_once_with("user-1", "cred-id-123")
    assert extra_kwargs["credentials"] == mock_creds
    assert mock_lock in locks


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
    from backend.executor.manager import _acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    # Key is present but value is None = user didn't select a file
    input_data = {"spreadsheet": None}

    with pytest.raises(ValueError, match="No file selected"):
        await _acquire_auto_credentials(
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
    from backend.executor.manager import _acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    # Key not present = connected from upstream block
    input_data = {}

    extra_kwargs, locks = await _acquire_auto_credentials(
        input_model=mock_input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    manager.acquire.assert_not_called()
    assert "credentials" not in extra_kwargs
    assert locks == []


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
    from backend.executor.manager import _acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    input_data = {"spreadsheet": google_drive_file_data["chained"]}

    extra_kwargs, locks = await _acquire_auto_credentials(
        input_model=mock_input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    manager.acquire.assert_not_called()
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
    from backend.executor.manager import _acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    input_data = {"spreadsheet": google_drive_file_data["missing_key"]}

    with pytest.raises(ValueError, match="Authentication missing"):
        await _acquire_auto_credentials(
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
    from backend.executor.manager import _acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    manager.acquire.side_effect = ValueError(
        "Credentials #cred-id-123 for user #user-2 not found"
    )
    input_data = {"spreadsheet": google_drive_file_data["valid"]}

    with pytest.raises(ValueError, match="not available in your account"):
        await _acquire_auto_credentials(
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
    from backend.executor.manager import _acquire_auto_credentials

    manager, _, _ = mock_creds_manager
    manager.acquire.side_effect = ValueError(
        "Credentials #cred-id-123 for user #user-1 not found"
    )
    input_data = {"spreadsheet": google_drive_file_data["valid"]}

    with pytest.raises(ValueError, match="not available in your account"):
        await _acquire_auto_credentials(
            input_model=mock_input_model,
            input_data=input_data,
            creds_manager=manager,
            user_id="user-1",
        )


@pytest.mark.asyncio
async def test_auto_credentials_lock_appended(
    mocker: MockerFixture,
    google_drive_file_data,
    mock_input_model,
    mock_creds_manager,
):
    """Lock from acquire() should be included in returned locks list."""
    from backend.executor.manager import _acquire_auto_credentials

    manager, _, mock_lock = mock_creds_manager
    input_data = {"spreadsheet": google_drive_file_data["valid"]}

    extra_kwargs, locks = await _acquire_auto_credentials(
        input_model=mock_input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    assert len(locks) == 1
    assert locks[0] is mock_lock


@pytest.mark.asyncio
async def test_auto_credentials_multiple_fields(
    mocker: MockerFixture,
    mock_creds_manager,
):
    """When there are multiple auto_credentials fields, only valid ones should acquire."""
    from backend.executor.manager import _acquire_auto_credentials

    manager, mock_creds, mock_lock = mock_creds_manager

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

    extra_kwargs, locks = await _acquire_auto_credentials(
        input_model=input_model,
        input_data=input_data,
        creds_manager=manager,
        user_id="user-1",
    )

    # Only the first field should have acquired credentials
    manager.acquire.assert_called_once_with("user-1", "cred-id-123")
    assert "credentials" in extra_kwargs
    assert "credentials2" not in extra_kwargs
    assert len(locks) == 1
