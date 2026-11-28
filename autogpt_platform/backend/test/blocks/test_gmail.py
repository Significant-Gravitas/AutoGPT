import base64
from email import message_from_bytes
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.google.gmail import (
    Attachment,
    GmailForwardBlock,
    GmailReadBlock,
    HasRecipients,
    _build_reply_message,
    create_mime_message,
    validate_all_recipients,
    validate_email_recipients,
)
from backend.data.execution import ExecutionContext
from backend.util.type import MediaFileType


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

        with patch("backend.blocks.google.gmail.BoundedHTML2Text") as mock_html2text:
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

        with patch("backend.blocks.google.gmail.BoundedHTML2Text") as mock_html2text:
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

        with patch(
            "backend.blocks.google.gmail.BoundedHTML2Text", side_effect=ImportError
        ):
            result = await self.gmail_block._get_email_body(msg, self.mock_service)
            assert result == html_text

    @pytest.mark.asyncio
    async def test_html_that_expands_when_converted_is_bounded(self):
        html_text = "<p>first words</p>" + "<ul>" * 20_000 + "<pre>" + "\n" * 80_000

        msg = {
            "id": "test_msg_expanding",
            "payload": {
                "mimeType": "text/html",
                "body": {"data": self._encode_base64(html_text)},
            },
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result.startswith("first words")
        assert len(result) <= 10 * len(html_text)

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

    def _part(self, mime_type: str, raw: bytes, content_type: str) -> dict:
        """A part as Gmail returns it: its headers, and its bytes as they were sent."""
        return {
            "mimeType": mime_type,
            "headers": [{"name": "Content-Type", "value": content_type}],
            "body": {"data": base64.urlsafe_b64encode(raw).decode("ascii")},
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text, content_type",
        [
            # Outlook's default for Western text: a curly apostrophe and a euro sign
            ("It’s €1.250 – due Friday.", "text/plain; charset=windows-1252"),
            ("La réunion est déplacée à jeudi.", 'text/plain; charset="ISO-8859-1"'),
            ("明日の会議は10時からです。", "text/plain; charset=ISO-2022-JP"),
            ("Привет, встреча завтра.", "text/plain; charset=koi8-r"),
            ("Grüße aus Köln.", "text/plain; charset=UTF-8"),
        ],
    )
    async def test_plain_body_is_decoded_with_its_charset(self, text, content_type):
        charset = content_type.split("charset=")[1].strip('"')
        msg = {
            "id": "test_msg_charset",
            "payload": self._part("text/plain", text.encode(charset), content_type),
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == text

    @pytest.mark.asyncio
    async def test_multipart_alternative_in_a_legacy_charset(self):
        text = "It’s attached, Anna."
        msg = {
            "id": "test_msg_cp1252",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [{"name": "Content-Type", "value": "multipart/alternative"}],
                "parts": [
                    self._part(
                        "text/plain",
                        text.encode("cp1252"),
                        "text/plain; charset=windows-1252",
                    ),
                    self._part(
                        "text/html",
                        f"<p>{text}</p>".encode("cp1252"),
                        "text/html; charset=windows-1252",
                    ),
                ],
            },
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == text

    @pytest.mark.asyncio
    async def test_html_body_is_decoded_with_its_charset(self):
        msg = {
            "id": "test_msg_html_latin1",
            "payload": self._part(
                "text/html",
                "<p>Café crème à 3€</p>".encode("iso-8859-15"),
                "text/html; charset=iso-8859-15",
            ),
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert "Café crème à 3€" in result

    @pytest.mark.asyncio
    async def test_body_stored_as_attachment_is_decoded_with_its_charset(self):
        text = "Größe: groß"
        msg = {
            "id": "test_msg_attachment_latin1",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "Content-Type", "value": "text/plain; charset=latin-1"}
                ],
                "body": {"attachmentId": "attachment_latin1"},
            },
        }
        self.mock_service.users().messages().attachments().get().execute.return_value = {
            "data": base64.urlsafe_b64encode(text.encode("latin-1")).decode("ascii")
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "content_type",
        ["text/plain; charset=UTF-8", "text/plain; charset=x-unknown", "text/plain"],
    )
    async def test_wrong_or_missing_charset_keeps_the_body(self, content_type):
        # windows-1252 bytes under a wrong, unknown or missing label
        msg = {
            "id": "test_msg_wrong_charset",
            "payload": self._part(
                "text/plain", "It’s due Friday.".encode("cp1252"), content_type
            ),
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "It\ufffds due Friday."

    @pytest.mark.asyncio
    @pytest.mark.parametrize("charset", ["undefined", "idna", "bad\x00name"])
    async def test_unusable_charset_label_falls_back_to_utf8(self, charset):
        msg = {
            "id": "test_msg_unusable_charset",
            "payload": self._part(
                "text/plain",
                "Grüße".encode("utf-8"),
                f'text/plain; charset="{charset}"',
            ),
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "Grüße"

    @pytest.mark.asyncio
    async def test_unparseable_charset_parameter_falls_back_to_utf8(self):
        msg = {
            "id": "test_msg_rfc2231_charset",
            "payload": self._part(
                "text/plain",
                "Grüße".encode("utf-8"),
                "text/plain; charset*=bad%00name''x",
            ),
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "Grüße"

    @pytest.mark.asyncio
    async def test_mixed_rfc2231_charset_continuations_fall_back_to_utf8(self):
        msg = {
            "id": "test_msg_rfc2231_continuations",
            "payload": self._part(
                "text/plain",
                "Grüße".encode("utf-8"),
                "text/plain; charset*=utf-8''utf-8; charset*0*=utf-8''utf-8",
            ),
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "Grüße"

    @pytest.mark.asyncio
    async def test_overlong_content_type_is_not_parsed(self):
        content_type = "text/plain; charset=windows-1252; " + "x=y; " * 400
        msg = {
            "id": "test_msg_long_header",
            "payload": self._part("text/plain", "Grüße".encode("utf-8"), content_type),
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "Grüße"

    @pytest.mark.asyncio
    async def test_binary_attachment_is_not_read_as_the_body(self):
        msg = {
            "id": "test_msg_scan_to_email",
            "payload": {
                "mimeType": "multipart/mixed",
                "parts": [
                    {"mimeType": "text/plain", "body": {"size": 0}},
                    {
                        "mimeType": "application/pdf",
                        "filename": "scan.pdf",
                        "body": {"attachmentId": "pdf_attachment"},
                    },
                ],
            },
        }
        self.mock_service.users().messages().attachments().get().execute.return_value = {
            "data": base64.urlsafe_b64encode(b"%PDF-1.4\n\xe2\xe3\xcf\xd3").decode()
        }

        result = await self.gmail_block._get_email_body(msg, self.mock_service)
        assert result == "This email does not contain a readable body."

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

    def test_address_at_the_length_limit(self):
        validate_email_recipients(["a" * (254 - len("@example.com")) + "@example.com"])

    def test_address_over_the_length_limit(self):
        with pytest.raises(ValueError, match="Invalid email address"):
            validate_email_recipients(
                ["a" * (255 - len("@example.com")) + "@example.com"]
            )

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


class TestValidateAllRecipients:
    """Test cases for validate_all_recipients."""

    def test_valid_all_fields(self):
        data = cast(
            HasRecipients,
            SimpleNamespace(to=["a@b.com"], cc=["c@d.com"], bcc=["e@f.com"]),
        )
        validate_all_recipients(data)

    def test_invalid_to_raises(self):
        data = cast(HasRecipients, SimpleNamespace(to=["bad"], cc=[], bcc=[]))
        with pytest.raises(ValueError, match="'to'"):
            validate_all_recipients(data)

    def test_invalid_cc_raises(self):
        data = cast(HasRecipients, SimpleNamespace(to=["a@b.com"], cc=["bad"], bcc=[]))
        with pytest.raises(ValueError, match="'cc'"):
            validate_all_recipients(data)

    def test_invalid_bcc_raises(self):
        data = cast(
            HasRecipients,
            SimpleNamespace(to=["a@b.com"], cc=["c@d.com"], bcc=["bad"]),
        )
        with pytest.raises(ValueError, match="'bcc'"):
            validate_all_recipients(data)

    def test_empty_cc_bcc_skipped(self):
        data = cast(HasRecipients, SimpleNamespace(to=["a@b.com"], cc=[], bcc=[]))
        validate_all_recipients(data)


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
        exec_ctx = cast(ExecutionContext, SimpleNamespace(graph_exec_id="test-exec-id"))

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
        exec_ctx = cast(ExecutionContext, SimpleNamespace(graph_exec_id="test-exec-id"))

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
        exec_ctx = cast(ExecutionContext, SimpleNamespace(graph_exec_id="test-exec-id"))

        # Should succeed without raising
        result = await create_mime_message(input_data, exec_ctx)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_attachments_are_limited_in_total(self, tmp_path):
        (tmp_path / "pad.bin").write_bytes(b"x" * 40)
        input_data = SimpleNamespace(
            to=["user@example.com"],
            cc=[],
            bcc=[],
            subject="Test",
            body="Hello",
            attachments=["pad.bin"] * 50,
        )
        exec_ctx = cast(ExecutionContext, SimpleNamespace(graph_exec_id="test-exec-id"))
        store = AsyncMock(return_value="pad.bin")

        with (
            patch("backend.blocks.google.gmail.store_media_file", new=store),
            patch(
                "backend.blocks.google.gmail.get_exec_file_path",
                side_effect=lambda _, path: str(tmp_path / path),
            ),
            patch("backend.util.file.MAX_TOTAL_FILE_SIZE_BYTES", 100),
            pytest.raises(ValueError, match="Files too large"),
        ):
            await create_mime_message(input_data, exec_ctx)

        assert store.await_count == 3


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
        exec_ctx = cast(ExecutionContext, SimpleNamespace(graph_exec_id="test-exec-id"))

        with pytest.raises(ValueError, match="Invalid email address"):
            await _build_reply_message(mock_service, input_data, exec_ctx)

    @pytest.mark.asyncio
    async def test_attachments_are_limited_in_total(self, tmp_path):
        (tmp_path / "pad.bin").write_bytes(b"x" * 40)
        mock_service = Mock()
        mock_service.users().messages().get().execute.return_value = {
            "threadId": "thread-1",
            "payload": {"headers": [{"name": "From", "value": "a@example.com"}]},
        }
        input_data = SimpleNamespace(
            parentMessageId="msg-1",
            to=["user@example.com"],
            cc=[],
            bcc=[],
            subject="",
            body="Reply body",
            replyAll=False,
            content_type=None,
            attachments=["pad.bin"] * 50,
        )
        exec_ctx = cast(ExecutionContext, SimpleNamespace(graph_exec_id="test-exec-id"))
        store = AsyncMock(return_value="pad.bin")

        with (
            patch("backend.blocks.google.gmail.store_media_file", new=store),
            patch(
                "backend.blocks.google.gmail.get_exec_file_path",
                side_effect=lambda _, path: str(tmp_path / path),
            ),
            patch("backend.util.file.MAX_TOTAL_FILE_SIZE_BYTES", 100),
            pytest.raises(ValueError, match="Files too large"),
        ):
            await _build_reply_message(mock_service, input_data, exec_ctx)

        assert store.await_count == 3


class TestForwardMessageValidation:
    """Test that _forward_message() raises ValueError for invalid recipients."""

    @staticmethod
    def _make_input(
        to: list[str] | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> "GmailForwardBlock.Input":
        mock = Mock(spec=GmailForwardBlock.Input)
        mock.messageId = "m1"
        mock.to = to or []
        mock.cc = cc or []
        mock.bcc = bcc or []
        mock.subject = ""
        mock.forwardMessage = "FYI"
        mock.includeAttachments = False
        mock.content_type = None
        mock.additionalAttachments = []
        mock.credentials = None
        return mock

    @staticmethod
    def _exec_ctx():
        return ExecutionContext(user_id="u1", graph_exec_id="g1")

    @staticmethod
    def _mock_service():
        """Build a mock Gmail service that returns a parent message."""
        parent_message = {
            "id": "m1",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Original subject"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "me@example.com"},
                    {"name": "Date", "value": "Mon, 31 Mar 2026 00:00:00 +0000"},
                ],
                "mimeType": "text/plain",
                "body": {
                    "data": base64.urlsafe_b64encode(b"Hello world").decode(),
                },
                "parts": [],
            },
        }
        svc = Mock()
        svc.users().messages().get().execute.return_value = parent_message
        return svc

    @pytest.mark.asyncio
    async def test_forward_carries_the_decoded_body(self):
        text = "La réunion est déplacée à jeudi."
        svc = self._mock_service()
        payload = svc.users().messages().get().execute.return_value["payload"]
        payload["headers"].append(
            {"name": "Content-Type", "value": "text/plain; charset=ISO-8859-1"}
        )
        payload["body"]["data"] = base64.urlsafe_b64encode(
            text.encode("iso-8859-1")
        ).decode()

        await GmailForwardBlock()._forward_message(
            svc, self._make_input(to=["valid@example.com"]), self._exec_ctx()
        )

        raw = svc.users().messages().send.call_args.kwargs["body"]["raw"]
        sent = message_from_bytes(base64.urlsafe_b64decode(raw))
        body = next(p for p in sent.walk() if p.get_content_type() == "text/plain")
        assert text in body.get_payload(decode=True).decode(body.get_content_charset())

    @pytest.mark.asyncio
    async def test_invalid_to_raises(self):
        block = GmailForwardBlock()
        with pytest.raises(ValueError, match="Invalid email address.*'to'"):
            await block._forward_message(
                self._mock_service(),
                self._make_input(to=["bad-addr"]),
                self._exec_ctx(),
            )

    @pytest.mark.asyncio
    async def test_invalid_cc_raises(self):
        block = GmailForwardBlock()
        with pytest.raises(ValueError, match="Invalid email address.*'cc'"):
            await block._forward_message(
                self._mock_service(),
                self._make_input(to=["valid@example.com"], cc=["not-valid"]),
                self._exec_ctx(),
            )

    @pytest.mark.asyncio
    async def test_invalid_bcc_raises(self):
        block = GmailForwardBlock()
        with pytest.raises(ValueError, match="Invalid email address.*'bcc'"):
            await block._forward_message(
                self._mock_service(),
                self._make_input(to=["valid@example.com"], bcc=["nope"]),
                self._exec_ctx(),
            )

    @pytest.mark.asyncio
    async def test_additional_attachments_are_limited_in_total(self, tmp_path):
        (tmp_path / "pad.bin").write_bytes(b"x" * 40)
        input_data = self._make_input(to=["valid@example.com"])
        input_data.additionalAttachments = [MediaFileType("pad.bin")] * 50
        store = AsyncMock(return_value="pad.bin")

        with (
            patch("backend.blocks.google.gmail.store_media_file", new=store),
            patch(
                "backend.blocks.google.gmail.get_exec_file_path",
                side_effect=lambda _, path: str(tmp_path / path),
            ),
            patch("backend.util.file.MAX_TOTAL_FILE_SIZE_BYTES", 100),
            pytest.raises(ValueError, match="Files too large"),
        ):
            await GmailForwardBlock()._forward_message(
                self._mock_service(), input_data, self._exec_ctx()
            )

        assert store.await_count == 3

    @pytest.mark.asyncio
    async def test_the_original_attachments_count_toward_the_total(self):
        block = GmailForwardBlock()
        input_data = self._make_input(to=["valid@example.com"])
        input_data.includeAttachments = True
        originals = [
            Attachment(
                filename=f"a{i}.bin",
                content_type="application/octet-stream",
                size=60,
                attachment_id=f"att-{i}",
            )
            for i in range(5)
        ]
        download = AsyncMock(return_value=b"x" * 60)

        with (
            patch.object(block, "_get_attachments", AsyncMock(return_value=originals)),
            patch.object(block, "download_attachment", download),
            patch("backend.util.file.MAX_TOTAL_FILE_SIZE_BYTES", 100),
            pytest.raises(ValueError, match="Files too large"),
        ):
            await block._forward_message(
                self._mock_service(), input_data, self._exec_ctx()
            )

        assert download.await_count == 1
