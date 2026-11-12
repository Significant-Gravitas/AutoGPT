from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr
from youtube_transcript_api._errors import NoTranscriptFound
from youtube_transcript_api._transcripts import FetchedTranscript, Transcript
from youtube_transcript_api.proxies import WebshareProxyConfig

from backend.blocks.youtube import TEST_CREDENTIALS, TranscribeYoutubeVideoBlock
from backend.data.model import UserPasswordCredentials
from backend.integrations.providers import ProviderName


class TestTranscribeYoutubeVideoBlock:
    """Test cases for TranscribeYoutubeVideoBlock."""

    def setup_method(self):
        """Set up test fixtures."""
        self.youtube_block = TranscribeYoutubeVideoBlock()
        self.credentials = TEST_CREDENTIALS

    @pytest.mark.parametrize(
        "url, video_id",
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dtUqwMu3e-g", "dtUqwMu3e-g"),
            ("https://www.youtube.com/shorts/dtUqwMu3e-g?feature=share", "dtUqwMu3e-g"),
            ("https://m.youtube.com/watch?v=aqz-KE-bpKQ&pp=ygUF", "aqz-KE-bpKQ"),
            ("https://www.youtube.com/live/aqz-KE-bpKQ?si=LiveNight", "aqz-KE-bpKQ"),
            ("https://youtube.com/live/aqz-KE-bpKQ", "aqz-KE-bpKQ"),
            ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://YouTube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("  https://youtu.be/dQw4w9WgXcQ  ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ/", "dQw4w9WgXcQ"),
        ],
    )
    def test_extract_video_id(self, url, video_id):
        assert self.youtube_block.extract_video_id(url) == video_id

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "https://www.youtube.com/watch?list=PL12345",
            "https://www.youtube.com/watch?v=",
            "https://www.youtube.com/watch",
            "https://youtu.be/",
            "https://youtu.be",
            "https://www.youtube.com/live/",
            "https://www.youtube.com/embed/",
            "https://www.youtube.com/v/",
            "https://www.youtube.com/shorts/",
            "https://notyoutube.com/watch?v=123",
            "https://youtube.com/channel/UC12345",
            "",
            "   ",
        ],
    )
    def test_extract_video_id_invalid_urls(self, invalid_url):
        with pytest.raises(ValueError, match="Invalid YouTube URL"):
            self.youtube_block.extract_video_id(invalid_url)

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_english_available(self, mock_api_class):
        """Test getting transcript when English is available."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_transcript = Mock(spec=FetchedTranscript)
        mock_api.fetch.return_value = mock_transcript

        result = self.youtube_block.get_transcript("test_video_id", self.credentials)

        assert result == mock_transcript
        mock_api_class.assert_called_once()
        proxy_config = mock_api_class.call_args[1]["proxy_config"]
        assert isinstance(proxy_config, WebshareProxyConfig)
        mock_api.fetch.assert_called_once_with(video_id="test_video_id")
        mock_api.list.assert_not_called()

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_with_custom_credentials(self, mock_api_class):
        """Test getting transcript with custom proxy credentials."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_transcript = Mock(spec=FetchedTranscript)
        mock_api.fetch.return_value = mock_transcript

        credentials = UserPasswordCredentials(
            provider=ProviderName.WEBSHARE_PROXY,
            username=SecretStr("custom_user"),
            password=SecretStr("custom_pass"),
        )

        result = self.youtube_block.get_transcript("test_video_id", credentials)

        assert result == mock_transcript
        mock_api_class.assert_called_once()
        proxy_config = mock_api_class.call_args[1]["proxy_config"]
        assert isinstance(proxy_config, WebshareProxyConfig)
        assert proxy_config.proxy_username == "custom_user"
        assert proxy_config.proxy_password == "custom_pass"
        mock_api.fetch.assert_called_once_with(video_id="test_video_id")
        mock_api.list.assert_not_called()

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_fallback_to_first_available(self, mock_api_class):
        """Test fallback to first available language when English is not available."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_transcript_list = Mock()
        mock_transcript_hu = Mock(spec=Transcript)
        mock_fetched_transcript = Mock(spec=FetchedTranscript)
        mock_transcript_hu.fetch.return_value = mock_fetched_transcript

        mock_transcript_list._manually_created_transcripts = {}
        mock_transcript_list._generated_transcripts = {"hu": mock_transcript_hu}

        mock_api.fetch.side_effect = NoTranscriptFound(
            "test_video_id", ("en",), mock_transcript_list
        )
        mock_api.list.return_value = mock_transcript_list

        result = self.youtube_block.get_transcript("test_video_id", self.credentials)

        assert result == mock_fetched_transcript
        mock_api_class.assert_called_once()
        mock_api.fetch.assert_called_once_with(video_id="test_video_id")
        mock_api.list.assert_called_once_with("test_video_id")
        mock_transcript_hu.fetch.assert_called_once()

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_prefers_manually_created(self, mock_api_class):
        """Test that manually created transcripts are preferred over generated ones."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_transcript_list = Mock()
        mock_transcript_manual = Mock(spec=Transcript)
        mock_transcript_generated = Mock(spec=Transcript)
        mock_fetched_manual = Mock(spec=FetchedTranscript)
        mock_transcript_manual.fetch.return_value = mock_fetched_manual

        mock_transcript_list._manually_created_transcripts = {
            "es": mock_transcript_manual
        }
        mock_transcript_list._generated_transcripts = {"hu": mock_transcript_generated}

        mock_api.fetch.side_effect = NoTranscriptFound(
            "test_video_id", ("en",), mock_transcript_list
        )
        mock_api.list.return_value = mock_transcript_list

        result = self.youtube_block.get_transcript("test_video_id", self.credentials)

        assert result == mock_fetched_manual
        mock_api_class.assert_called_once()
        mock_transcript_manual.fetch.assert_called_once()
        mock_transcript_generated.fetch.assert_not_called()

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_no_transcripts_available(self, mock_api_class):
        """Test that exception is re-raised when no transcripts are available at all."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_transcript_list = Mock()
        mock_transcript_list._manually_created_transcripts = {}
        mock_transcript_list._generated_transcripts = {}

        original_exception = NoTranscriptFound(
            "test_video_id", ("en",), mock_transcript_list
        )
        mock_api.fetch.side_effect = original_exception
        mock_api.list.return_value = mock_transcript_list

        with pytest.raises(NoTranscriptFound):
            self.youtube_block.get_transcript("test_video_id", self.credentials)
        mock_api_class.assert_called_once()

    @patch("backend.blocks.youtube.TextFormatter")
    def test_format_transcript(self, mock_formatter_class):
        """Test transcript formatting."""
        mock_formatter = Mock()
        mock_formatter_class.return_value = mock_formatter
        mock_formatter.format_transcript.return_value = "Formatted transcript text"

        mock_transcript = Mock(spec=FetchedTranscript)
        result = self.youtube_block.format_transcript(mock_transcript)

        assert result == "Formatted transcript text"
        mock_formatter.format_transcript.assert_called_once_with(mock_transcript)

    @pytest.mark.asyncio
    @patch.object(TranscribeYoutubeVideoBlock, "get_transcript")
    @patch.object(TranscribeYoutubeVideoBlock, "format_transcript")
    async def test_run_success(self, mock_format, mock_get_transcript):
        """Test successful execution of run method."""
        mock_transcript = Mock(spec=FetchedTranscript)
        mock_get_transcript.return_value = mock_transcript
        mock_format.return_value = "Never gonna give you up"

        input_data = TranscribeYoutubeVideoBlock.Input(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            credentials=self.credentials,
        )

        outputs = [
            out
            async for out in self.youtube_block.run(
                input_data, credentials=self.credentials
            )
        ]

        assert ("video_id", "dQw4w9WgXcQ") in outputs
        assert ("transcript", "Never gonna give you up") in outputs
        mock_get_transcript.assert_called_once_with("dQw4w9WgXcQ", self.credentials)
        mock_format.assert_called_once_with(transcript=mock_transcript)

    @pytest.mark.asyncio
    async def test_run_invalid_url_yields_error(self):
        """Test that invalid URL in run method yields error output."""
        input_data = TranscribeYoutubeVideoBlock.Input(
            youtube_url="https://notyoutube.com/watch?v=123",
            credentials=self.credentials,
        )

        outputs = [
            out
            async for out in self.youtube_block.run(
                input_data, credentials=self.credentials
            )
        ]

        assert len(outputs) == 1
        key, error_msg = outputs[0]
        assert key == "error"
        assert "Invalid YouTube URL" in error_msg

    @pytest.mark.asyncio
    @patch.object(TranscribeYoutubeVideoBlock, "get_transcript")
    async def test_run_transcript_error_yields_error(self, mock_get_transcript):
        """Test that transcript fetching failure in run method yields error output."""
        mock_get_transcript.side_effect = RuntimeError("Proxy connection failed")

        input_data = TranscribeYoutubeVideoBlock.Input(
            youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            credentials=self.credentials,
        )

        outputs = [
            out
            async for out in self.youtube_block.run(
                input_data, credentials=self.credentials
            )
        ]

        assert len(outputs) == 1
        key, error_msg = outputs[0]
        assert key == "error"
        assert "Proxy connection failed" in error_msg
