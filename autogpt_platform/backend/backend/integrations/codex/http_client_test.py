import base64
import json

from pydantic import SecretStr

from backend.integrations.codex.auth_bundle import CodexAuthBundleV1, CodexAuthTokensV1
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.http_client import (
    _parse_model,
    account_id_for,
    account_snapshot,
    parse_rate_limits,
)


def _jwt(payload: dict[str, object]) -> str:
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )
    return f"header.{encoded}.signature"


def _credentials_with_identity_only_in_id_token():
    id_token = _jwt(
        {
            "email": "person@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_plan_type": "pro",
                "chatgpt_account_id": "account-from-id-token",
            },
        }
    )
    access_token = _jwt({"exp": 2_000_000_000})
    bundle = CodexAuthBundleV1(
        tokens=CodexAuthTokensV1(
            id_token=SecretStr(id_token),
            access_token=SecretStr(access_token),
            refresh_token=SecretStr("refresh-token"),
        ),
        codex_runtime_version="http",
    )
    return credentials_from_bundle(bundle)


def test_account_identity_comes_from_the_id_token() -> None:
    credentials = _credentials_with_identity_only_in_id_token()

    assert account_id_for(credentials) == "account-from-id-token"
    assert account_snapshot(credentials).model_dump() == {
        "connected": True,
        "requires_openai_auth": False,
        "account_type": "chatgpt",
        "email": "person@example.com",
        "plan_type": "pro",
    }


def _model(**over: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "slug": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "visibility": "list",
        "priority": 1,
        "default_reasoning_level": "low",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "…"},
            {"effort": "medium", "description": "…"},
        ],
        "input_modalities": ["text", "image"],
    }
    entry.update(over)
    return entry


# --------------------------------------------------------------------------- #
# Model catalog
# --------------------------------------------------------------------------- #


def test_reasoning_levels_are_unwrapped_from_their_description_objects() -> None:
    parsed = _parse_model(_model())
    assert parsed is not None
    assert parsed.supported_reasoning_efforts == ["low", "medium"]


def test_unknown_reasoning_levels_are_dropped_rather_than_failing_the_model() -> None:
    """A new effort tier upstream must not take the whole catalog down."""
    parsed = _parse_model(
        _model(supported_reasoning_levels=[{"effort": "low"}, {"effort": "warp"}])
    )
    assert parsed is not None
    assert parsed.supported_reasoning_efforts == ["low"]


def test_an_unknown_default_effort_falls_back_to_a_supported_one() -> None:
    parsed = _parse_model(_model(default_reasoning_level="warp"))
    assert parsed is not None
    assert parsed.default_reasoning_effort == "low"


def test_hidden_is_driven_by_visibility() -> None:
    assert _parse_model(_model(visibility="hide")).hidden is True  # type: ignore[union-attr]
    assert _parse_model(_model(visibility="list")).hidden is False  # type: ignore[union-attr]


def test_an_entry_without_a_slug_is_skipped() -> None:
    assert _parse_model(_model(slug="")) is None
    assert _parse_model({"display_name": "orphan"}) is None
    assert _parse_model("not-a-dict") is None


def test_display_name_falls_back_to_the_slug() -> None:
    parsed = _parse_model(_model(display_name=None))
    assert parsed is not None
    assert parsed.display_name == "gpt-5.6-sol"


# --------------------------------------------------------------------------- #
# Rate limits
#
# These arrive as headers on every inference call, so a misparse silently
# misreports someone's remaining quota rather than raising.
# --------------------------------------------------------------------------- #


def test_credit_booleans_are_title_cased_on_the_wire() -> None:
    """Sent as "False"/"True" — truthiness on the raw string inverts the flag."""
    limits = parse_rate_limits(
        {
            "x-codex-credits-has-credits": "False",
            "x-codex-credits-unlimited": "True",
        }
    )
    assert limits.has_credits is False
    assert limits.unlimited_credits is True


def test_absent_credit_headers_stay_unknown_rather_than_becoming_false() -> None:
    limits = parse_rate_limits({})
    assert limits.has_credits is None
    assert limits.unlimited_credits is None


def test_a_zero_length_window_is_not_reported_as_a_real_window() -> None:
    """An inactive tier reports window-minutes=0; keeping it invents a limit."""
    limits = parse_rate_limits(
        {
            "x-codex-secondary-used-percent": "0",
            "x-codex-secondary-window-minutes": "0",
            "x-codex-secondary-reset-at": "",
        }
    )
    assert limits.secondary is None


def test_an_active_window_is_parsed_whole() -> None:
    limits = parse_rate_limits(
        {
            "x-codex-plan-type": "pro",
            "x-codex-active-limit": "premium",
            "x-codex-primary-used-percent": "3",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-primary-reset-at": "1788452827",
        }
    )
    assert limits.plan_type == "pro"
    assert limits.limit_id == "premium"
    assert limits.primary is not None
    assert limits.primary.used_percent == 3
    assert limits.primary.window_duration_mins == 10080
    assert limits.primary.resets_at == 1788452827


def test_header_lookup_is_case_insensitive() -> None:
    limits = parse_rate_limits({"X-Codex-Plan-Type": "pro"})
    assert limits.plan_type == "pro"


def test_a_zero_percent_window_is_kept_when_the_window_is_real() -> None:
    """0% used is a valid reading; only a zero-length *window* means absent."""
    limits = parse_rate_limits(
        {
            "x-codex-primary-used-percent": "0",
            "x-codex-primary-window-minutes": "300",
        }
    )
    assert limits.primary is not None
    assert limits.primary.used_percent == 0


def test_unparseable_numbers_do_not_raise() -> None:
    limits = parse_rate_limits(
        {
            "x-codex-primary-used-percent": "n/a",
            "x-codex-primary-window-minutes": "300",
        }
    )
    assert limits.primary is None
