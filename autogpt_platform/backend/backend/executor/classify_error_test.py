"""
Tests for ``classify_user_credentials_error`` in ``backend/executor/manager.py``.

This helper is the contract the SQL classifier in
``get_block_error_stats`` relies on: it decides whether a node failure
should be tagged ``error_type='user_credentials_invalid'``. The tag must
NOT be set when the failing credential is platform-managed, otherwise a
real platform incident (e.g. our managed key being rotated) gets silently
excluded from the block-error-rate alert.
"""

from pydantic import SecretStr
from pytest_mock import MockerFixture

from backend.data.model import APIKeyCredentials
from backend.executor.manager import classify_user_credentials_error
from backend.util.exceptions import (
    BlockExecutionError,
    BlockUserCredentialsInvalidError,
)


def _make_creds(*, is_managed: bool) -> APIKeyCredentials:
    return APIKeyCredentials(
        id="cred-123",
        provider="anthropic",
        api_key=SecretStr("sk-test"),
        title="user key",
        is_managed=is_managed,
    )


def _make_input_model(mocker: MockerFixture, cred_field_names: list[str]):
    input_model = mocker.MagicMock()
    input_model.get_credentials_fields.return_value = {
        name: mocker.MagicMock() for name in cred_field_names
    }
    return input_model


def test_user_credential_invalid_with_user_supplied_key_classifies(
    mocker: MockerFixture,
):
    """Typed error + user (non-managed) credential → tag as user error."""
    input_model = _make_input_model(mocker, ["credentials"])
    extra_kwargs = {
        "user_id": "u1",
        "credentials": _make_creds(is_managed=False),
    }
    err = BlockUserCredentialsInvalidError("invalid key", "TestBlock", "b1")
    assert classify_user_credentials_error(err, extra_kwargs, input_model) is True


def test_user_credential_invalid_with_managed_key_does_not_classify(
    mocker: MockerFixture,
):
    """Typed error + platform-managed credential → do NOT tag.

    Regression scenario: our managed key gets rotated, downstream block
    surfaces an invalid_api_key style error and raises the typed class.
    Tagging this would silently exclude it from the platform alert and
    on-call would never get paged for our own rotated platform key.
    """
    input_model = _make_input_model(mocker, ["credentials"])
    extra_kwargs = {
        "user_id": "u1",
        "credentials": _make_creds(is_managed=True),
    }
    err = BlockUserCredentialsInvalidError("invalid key", "TestBlock", "b1")
    assert classify_user_credentials_error(err, extra_kwargs, input_model) is False


def test_classifies_when_credential_field_is_not_named_credentials(
    mocker: MockerFixture,
):
    """Blocks like ClaudeCodeBlock declare ``anthropic_credentials`` rather
    than ``credentials``. The classifier must traverse ``get_credentials_fields()``
    rather than only checking the literal key ``credentials``."""
    input_model = _make_input_model(mocker, ["anthropic_credentials"])
    extra_kwargs = {
        "user_id": "u1",
        "anthropic_credentials": _make_creds(is_managed=False),
    }
    err = BlockUserCredentialsInvalidError("invalid key", "TestBlock", "b1")
    assert classify_user_credentials_error(err, extra_kwargs, input_model) is True


def test_does_not_classify_generic_exception(mocker: MockerFixture):
    """Substring matching on ``invalid_api_key`` would have false-positively
    excluded these; the typed signal does not."""
    input_model = _make_input_model(mocker, ["credentials"])
    extra_kwargs = {"credentials": _make_creds(is_managed=False)}
    err = RuntimeError("upstream returned: invalid_api_key (platform key rotated)")
    assert classify_user_credentials_error(err, extra_kwargs, input_model) is False


def test_does_not_classify_block_execution_error_supertype(mocker: MockerFixture):
    """Parent class must not classify — only the specific user-credentials subclass."""
    input_model = _make_input_model(mocker, ["credentials"])
    extra_kwargs = {"credentials": _make_creds(is_managed=False)}
    err = BlockExecutionError("output missing", "TestBlock", "b1")
    assert classify_user_credentials_error(err, extra_kwargs, input_model) is False


def test_does_not_classify_when_no_resolved_credential(mocker: MockerFixture):
    """If the block had no credential, there's no signal to classify on."""
    input_model = _make_input_model(mocker, ["credentials"])
    extra_kwargs = {"user_id": "u1"}
    err = BlockUserCredentialsInvalidError("invalid key", "TestBlock", "b1")
    assert classify_user_credentials_error(err, extra_kwargs, input_model) is False


def test_does_not_classify_when_any_credential_is_managed(mocker: MockerFixture):
    """Multi-credential block: if ANY resolved credential is managed, treat
    the failure as a platform issue (conservative — better to over-page than
    silence a real incident)."""
    input_model = _make_input_model(
        mocker, ["anthropic_credentials", "e2b_credentials"]
    )
    extra_kwargs = {
        "anthropic_credentials": _make_creds(is_managed=False),
        "e2b_credentials": _make_creds(is_managed=True),
    }
    err = BlockUserCredentialsInvalidError("invalid key", "TestBlock", "b1")
    assert classify_user_credentials_error(err, extra_kwargs, input_model) is False
