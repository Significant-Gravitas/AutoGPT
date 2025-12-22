"""Tests for the FieldSchemaExtra TypedDict definitions."""

import pytest

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
        extra: FieldSchemaExtra = {
            "placeholder": "test placeholder",
            "auto_credentials": {
                "provider": "google",
                "type": "oauth2",
            },
        }
        field = SchemaField(
            default="test",
            description="Test field",
            json_schema_extra=extra,
        )
        # SchemaField returns the default value, so we just verify it doesn't raise
        assert field == "test"
