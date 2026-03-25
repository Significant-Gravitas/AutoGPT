import base64
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from backend.blocks.google.gmail import (
    GmailReadBlock,
    _build_reply_message,
    create_mime_message,
    validate_email_recipients,
)


class TestGmailReadBlock:
    """Test cases for GmailReadBlock email body parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gmail_block = GmailReadBlock()
        self.mock_service = Mock()

    def _encode_base64(self, text: str) -> str:
        """Helper to encode text as base64 URL-safe."""
        return base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8")

    @pytest.mark.asyncio
    async def test_single_part_text_plain(self):
        """Test parsing single-part text/plain email."""
        body_text = "This is a plain text email body."
        msg = {
            "id": "test_msg_1",
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": self._encode_base64(body_text)},
            },
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == body_text

    @pytest.mark.asyncio
    async def test_multipart_alternative_plain_and_html(self):
        """Test parsing multipart/alternative with both plain and HTML parts."""
        plain_text = "This is the plain text version."
        html_text = "<html><body><p>This is the HTML version.</p></body></html>"

        msg = {
            "id": "test_msg_2",
            "payload": {
                "mimeType": "multipart/alternative",
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": self._encode_base64(plain_text)},
                    },
                    {
                        "mimeType": "text/html",
                        "body": {"data": self._encode_base64(html_text)},
                    },
                ],
            },
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        # Should prefer plain text over HTML
        assert result == plain_text

    @pytest.mark.asyncio
    async def test_html_only_email(self):
        """Test parsing HTML-only email with conversion to plain text."""
        html_text = (
            "<html><body><h1>Hello World</h1><p>This is HTML content.</p></body></html>"
        )

        msg = {
            "id": "test_msg_3",
            "payload": {
                "mimeType": "text/html",
                "body": {"data": self._encode_base64(html_text)},
            },
        }

        with patch("html2text.HTML2Text") as mock_html2text:
            mock_converter = Mock()
            mock_converter.handle.return_value = "Hello World\n\nThis is HTML content."
            mock_html2text.return_value = mock_converter

            result = await self.gmail_block._get_email_body(msg, self.mock_service)
            assert "Hello World" in result
            assert "This is HTML content" in result

    @pytest.mark.asyncio
    async def test_html_fallback_when_html2text_conversion_fails(self):
        """Fallback to raw HTML when html2text converter raises unexpectedly."""
        html_text = "<html><body><p>Broken <b>HTML</p></body></html>"

        msg = {
            "id": "test_msg_html_error",
            "payload": {
                "mimeType": "text/html",
                "body": {"data": self._encode_base64(html_text)},
            },
        }

        with patch("html2text.HTML2Text") as mock_html2text:
            mock_converter = Mock()
            mock_converter.handle.side_effect = ValueError("conversion failed")
            mock_html2text.return_value = mock_converter

            result = await self.gmail_block._get_email_body(msg, self.mock_service)
            assert result == html_text

    @pytest.mark.asyncio
    async def test_html_fallback_when_html2text_unavailable(self):
        """Test fallback to raw HTML when html2text is not available."""
        html_text = "<html><body><p>HTML content</p></body></html>"

        msg = {
            "id": "test_msg_4",
            "payload": {
                "mimeType": "text/html",
                "body": {"data": self._encode_base64(html_text)},
            },
        }

        with patch("html2text.HTML2Text", side_effect=ImportError):
            result = await self.gmail_block._get_email_body(msg, self.mock_service)
            assert result == html_text

    @pytest.mark.asyncio
    async def test_nested_multipart_structure(self):
        """Test parsing deeply nested multipart structure."""
        plain_text = "Nested plain text content."

        msg = {
            "id": "test_msg_5",
            "payload": {
                "mimeType": "multipart/mixed",
                "parts": [
                    {
                        "mimeType": "multipart/alternative",
                        "parts": [
                            {
                                "mimeType": "text/plain",
                                "body": {"data": self._encode_base64(plain_text)},
                            },
                        ],
                    },
                ],
            },
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == plain_text

    @pytest.mark.asyncio
    async def test_attachment_body_content(self):
        """Test parsing email where body is stored as attachment."""
        attachment_data = self._encode_base64("Body content from attachment.")

        msg = {
            "id": "test_msg_6",
            "payload": {
                "mimeType": "text/plain",
                "body": {"attachmentId": "attachment_123"},
            },
        }

        # Mock the attachment download
        self.mock_service.users().messages().attachments().get().execute.return_value = {
            "data": attachment_data
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "Body content from attachment."

    @pytest.mark.asyncio
    async def test_no_readable_body(self):
        """Test email with no readable body content."""
        msg = {
            "id": "test_msg_7",
            "payload": {
                "mimeType": "application/octet-stream",
                "body": {},
            },
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "This email does not contain a readable body."

    @pytest.mark.asyncio
    async def test_base64_padding_handling(self):
        """Test proper handling of base64 data with missing padding."""
        # Create base64 data with missing padding
        text = "Test content"
        encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8")
        # Remove padding
        encoded_no_padding = encoded.rstrip("=")

        result = self.gmail_block._decode_base64(encoded_no_padding)
        assert result == text

    @pytest.mark.asyncio
    async def test_recursion_depth_limit(self):
        """Test that recursion depth is properly limited."""

        # Create a deeply nested structure that would exceed the limit
        def create_nested_part(depth):
            if depth > 15:  # Exceed the limit of 10
                return {
                    "mimeType": "text/plain",
                    "body": {"data": self._encode_base64("Deep content")},
                }
            return {
                "mimeType": "multipart/mixed",
                "parts": [create_nested_part(depth + 1)],
            }

        msg = {
            "id": "test_msg_8",
            "payload": create_nested_part(0),
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        # Should return fallback message due to depth limit
        assert result == "This email does not contain a readable body."

    @pytest.mark.asyncio
    async def test_malformed_base64_handling(self):
        """Test handling of malformed base64 data."""
        result = self.gmail_block._decode_base64("invalid_base64_data!!!")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_data_handling(self):
        """Test handling of empty or None data."""
        assert self.gmail_block._decode_base64("") is None
        assert self.gmail_block._decode_base64(None) is None

    @pytest.mark.asyncio
    async def test_attachment_download_failure(self):
        """Test handling of attachment download failure."""
        msg = {
            "id": "test_msg_9",
            "payload": {
                "mimeType": "text/plain",
                "body": {"attachmentId": "invalid_attachment"},
            },
        }

        # Mock attachment download failure
        self.mock_service.users().messages().attachments().get().execute.side_effect = (
            Exception("Download failed")
        )

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "This email does not contain a readable body."


class TestValidateEmailRecipients:
    """Test cases for validate_email_recipients."""

    def test_valid_single_email(self):
        validate_email_recipients(["user@example.com"])

    def test_valid_multiple_emails(self):
        validate_email_recipients(["a@b.com", "x@y.org", "test@sub.domain.co"])

    def test_invalid_missing_at(self):
        with pytest.raises(ValueError, match="Invalid email address"):
            validate_email_recipients(["not-an-email"])

    def test_invalid_missing_domain_dot(self):
        with pytest.raises(ValueError, match="Invalid email address"):
            validate_email_recipients(["user@localhost"])

    def test_invalid_empty_string(self):
        with pytest.raises(ValueError, match="Invalid email address"):
            validate_email_recipients([""])

    def test_invalid_json_object_string(self):
        with pytest.raises(ValueError, match="Invalid email address"):
            validate_email_recipients(['{"email": "user@example.com"}'])

    def test_mixed_valid_and_invalid(self):
        with pytest.raises(ValueError, match="'bad-addr'"):
            validate_email_recipients(["good@example.com", "bad-addr"])

    def test_field_name_in_error(self):
        with pytest.raises(ValueError, match="'cc'"):
            validate_email_recipients(["nope"], field_name="cc")

    def test_whitespace_trimmed(self):
        validate_email_recipients(["  user@example.com  "])

    def test_empty_list_passes(self):
        validate_email_recipients([])


class TestCreateMimeMessageValidation:
    """Integration tests verifying validation hooks in create_mime_message()."""

    @pytest.mark.asyncio
    async def test_invalid_to_raises_before_mime_construction(self):
        """Invalid 'to' recipients should raise ValueError before any MIME work."""
        input_data = SimpleNamespace(
            to=["not-an-email"],
            cc=[],
            bcc=[],
            subject="Test",
            body="Hello",
            attachments=[],
        )
        exec_ctx = SimpleNamespace(graph_exec_id="test-exec-id")

        with pytest.raises(ValueError, match="Invalid email address"):
            await create_mime_message(input_data, exec_ctx)

    @pytest.mark.asyncio
    async def test_invalid_cc_raises_before_mime_construction(self):
        """Invalid 'cc' recipients should raise ValueError."""
        input_data = SimpleNamespace(
            to=["valid@example.com"],
            cc=["bad-addr"],
            bcc=[],
            subject="Test",
            body="Hello",
            attachments=[],
        )
        exec_ctx = SimpleNamespace(graph_exec_id="test-exec-id")

        with pytest.raises(ValueError, match="'cc'"):
            await create_mime_message(input_data, exec_ctx)

    @pytest.mark.asyncio
    async def test_valid_recipients_passes_validation(self):
        """Valid recipients should not raise during validation."""
        input_data = SimpleNamespace(
            to=["user@example.com"],
            cc=["other@example.com"],
            bcc=[],
            subject="Test",
            body="Hello",
            attachments=[],
        )
        exec_ctx = SimpleNamespace(graph_exec_id="test-exec-id")

        # Should succeed without raising
        result = await create_mime_message(input_data, exec_ctx)
        assert isinstance(result, str)


class TestBuildReplyMessageValidation:
    """Integration tests verifying validation hooks in _build_reply_message()."""

    @pytest.mark.asyncio
    async def test_invalid_to_raises_before_reply_construction(self):
        """Invalid 'to' in reply should raise ValueError before MIME work."""
        mock_service = Mock()
        mock_parent = {
            "threadId": "thread-1",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Original"},
                    {"name": "Message-ID", "value": "<msg@example.com>"},
                    {"name": "From", "value": "sender@example.com"},
                ]
            },
        }
        mock_service.users().messages().get().execute.return_value = mock_parent

        input_data = SimpleNamespace(
            parentMessageId="msg-1",
            to=["not-valid"],
            cc=[],
            bcc=[],
            subject="",
            body="Reply body",
            replyAll=False,
            attachments=[],
        )
        exec_ctx = SimpleNamespace(graph_exec_id="test-exec-id")

        with pytest.raises(ValueError, match="Invalid email address"):
            await _build_reply_message(mock_service, input_data, exec_ctx)
