"""Tests for platform bot linking API routes."""

import pytest
from fastapi import HTTPException

from backend.api.features.platform_linking.routes import (
    VALID_PLATFORMS,
    _validate_platform,
)


class TestValidatePlatform:
    def test_valid_platforms(self):
        for platform in VALID_PLATFORMS:
            # Should not raise
            _validate_platform(platform)

    def test_invalid_platform(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_platform("INVALID")
        assert exc_info.value.status_code == 400

    def test_lowercase_rejected(self):
        with pytest.raises(HTTPException):
            _validate_platform("discord")
