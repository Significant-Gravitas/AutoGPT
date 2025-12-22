"""Tests for the FieldSchemaExtra TypedDict definitions."""

import pytest
from typing import Any
from pydantic import BaseModel

from backend.data.model import (
    AutoCredentialsConfig,
    FieldSchemaExtra,
    GoogleDrivePickerConfig,
    SchemaField,
)


class TestFieldSchemaExtra:
    """Test cases for FieldSchemaExtra TypedDict."""

    def test_field_schema_extra_basic_fields(self):
        """Test that basic FieldSchemaExtra fields are typed correctly."""
        schema: FieldSchemaExtra = {
            "placeholder": "Enter value...",
            "secret": True,
            "advanced": False,
            "hidden": False,
            "format": "text",
        }
        assert schema["placeholder"] == "Enter value..."
        assert schema["secret"] is True
        assert schema["advanced"] is False

    def test_field_schema_extra_credentials_fields(self):
        """Test credential-related FieldSchemaExtra fields."""
        schema: FieldSchemaExtra = {
            "credentials_provider": ["google", "github"],
            "credentials_types": ["oauth2", "api_key"],
            "credentials_scopes": ["read", "write"],
        }
        assert "google" in schema["credentials_provider"]
        assert "oauth2" in schema["credentials_types"]

    def test_auto_credentials_config(self):
        """Test AutoCredentialsConfig TypedDict."""
        config: AutoCredentialsConfig = {
            "provider": "google",
            "type": "oauth2",
            "scopes": ["drive.readonly"],
            "kwarg_name": "credentials",
        }
        assert config["provider"] == "google"
        assert config["type"] == "oauth2"

    def test_google_drive_picker_config(self):
        """Test GoogleDrivePickerConfig TypedDict."""
        config: GoogleDrivePickerConfig = {
            "allowed_mime_types": ["application/pdf", "image/png"],
            "auto_credentials": {
                "provider": "google",
                "type": "oauth2",
                "scopes": ["drive.readonly"],
            },
        }
        assert "application/pdf" in config["allowed_mime_types"]

    def test_schema_field_with_typed_extra(self):
        """Test SchemaField accepts FieldSchemaExtra typed dict."""
        extra: dict[str, Any] = {
            "placeholder": "test placeholder",
            "auto_credentials": {
                "provider": "google",
                "type": "oauth2",
            },
        }

        # Create a test model to use SchemaField in context
        class TestModel(BaseModel):
            test_field: str = SchemaField(
                default="test",
                description="Test field",
                json_schema_extra=extra,
            )

        # Verify the field was created with correct schema
        field_info = TestModel.model_fields["test_field"]
        assert field_info.default == "test"
        assert field_info.description == "Test field"
        # json_schema_extra can be a dict or callable, check if it's a dict
        if isinstance(field_info.json_schema_extra, dict):
            assert field_info.json_schema_extra.get("placeholder") == "test placeholder"
