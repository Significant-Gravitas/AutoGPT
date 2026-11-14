"""Admission is between the one candidate lookup and any expensive verification."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from prisma.enums import APIKeyStatus

from backend.data.auth import api_key


@pytest.fixture
def candidate(mocker):
    generated = api_key.keysmith.generate_key()
    candidate = api_key.APIKeyInfoWithHash(
        id="key-1",
        name="Key",
        head=generated.head,
        tail=generated.tail,
        status=APIKeyStatus.ACTIVE,
        user_id="owner-1",
        scopes=[],
        created_at=datetime.now(UTC),
        hash=generated.hash,
        salt=generated.salt,
    )
    lookup = mocker.patch.object(
        api_key, "get_active_api_keys_by_head", return_value=[candidate]
    )
    mocker.patch.object(api_key, "_record_use")
    return generated.key, lookup


async def test_admission_runs_after_lookup_once_and_before_hashing(candidate, mocker):
    plaintext, lookup = candidate
    verify = mocker.spy(api_key.keysmith, "verify_key")

    async def admit():
        lookup.assert_awaited_once_with(plaintext[:8], tail=plaintext[-8:])
        verify.assert_not_called()

    admission = AsyncMock(side_effect=admit)
    result = await api_key.validate_api_key(plaintext, before_verify=admission)

    assert result is not None and result.user_id == "owner-1"
    admission.assert_awaited_once()
    lookup.assert_awaited_once()
    assert verify.call_count == 1


async def test_admission_rejection_propagates_without_hashing(candidate, mocker):
    rejection = ValueError("admission denied")
    admission = AsyncMock(side_effect=rejection)
    verify = mocker.spy(api_key.keysmith, "verify_key")

    with pytest.raises(ValueError) as raised:
        await api_key.validate_api_key(candidate[0], before_verify=admission)

    assert raised.value is rejection
    admission.assert_awaited_once()
    verify.assert_not_called()


async def test_no_candidates_skip_admission_and_hashing(candidate, mocker):
    mocker.patch.object(api_key, "get_active_api_keys_by_head", return_value=[])
    admission = AsyncMock()
    verify = mocker.spy(api_key.keysmith, "verify_key")

    assert await api_key.validate_api_key(candidate[0], before_verify=admission) is None

    admission.assert_not_called()
    verify.assert_not_called()


async def test_lookup_failure_preserves_error_without_admission(candidate, mocker):
    mocker.patch.object(api_key, "get_active_api_keys_by_head", side_effect=OSError())
    admission = AsyncMock()

    with pytest.raises(RuntimeError, match="Failed to validate API key"):
        await api_key.validate_api_key(candidate[0], before_verify=admission)

    admission.assert_not_called()


async def test_hash_failure_preserves_error_after_admission(candidate, mocker):
    mocker.patch.object(api_key.keysmith, "verify_key", side_effect=ValueError())
    admission = AsyncMock()

    with pytest.raises(RuntimeError, match="Failed to validate API key"):
        await api_key.validate_api_key(candidate[0], before_verify=admission)

    admission.assert_awaited_once()
