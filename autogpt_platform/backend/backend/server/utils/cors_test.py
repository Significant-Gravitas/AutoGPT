import pytest

from backend.server.utils.cors import build_cors_params
from backend.util.settings import AppEnvironment


def test_build_cors_params_splits_regex_patterns() -> None:
    origins = [
        "https://app.example.com",
        "regex:https://.*\\.example\\.com",
    ]

    result = build_cors_params(origins, AppEnvironment.LOCAL)

    assert result["allow_origins"] == ["https://app.example.com"]
    assert result["allow_origin_regex"] == "^(?:https://.*\\.example\\.com)$"


def test_build_cors_params_combines_multiple_regex_patterns() -> None:
    origins = [
        "regex:https://alpha.example.com",
        "regex:https://beta.example.com",
    ]

    result = build_cors_params(origins, AppEnvironment.DEVELOPMENT)

    assert result["allow_origins"] == []
    assert result["allow_origin_regex"] == (
        "^(?:https://alpha.example.com|https://beta.example.com)$"
    )


def test_build_cors_params_blocks_localhost_literal_in_production() -> None:
    with pytest.raises(ValueError):
        build_cors_params(["http://localhost:3000"], AppEnvironment.PRODUCTION)


def test_build_cors_params_blocks_localhost_regex_in_production() -> None:
    with pytest.raises(ValueError):
        build_cors_params(
            ["regex:https://.*localhost.*"], AppEnvironment.PRODUCTION
        )

