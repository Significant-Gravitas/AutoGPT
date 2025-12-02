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
    """Test cases for TranscribeYoutubeVideoBlock language fallback functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.youtube_block = TranscribeYoutubeVideoBlock()
        self.credentials = TEST_CREDENTIALS

    def test_extract_video_id_standard_url(self):
        """Test extracting video ID from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = self.youtube_block.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_short_url(self):
        """Test extracting video ID from shortened youtu.be URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = self.youtube_block.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_video_id_embed_url(self):
        """Test extracting video ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        video_id = self.youtube_block.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_english_available(self, mock_api_class):
        """Test getting transcript when English is available."""
        # Setup mock
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_transcript = Mock(spec=FetchedTranscript)
        mock_api.fetch.return_value = mock_transcript

        # Execute
        result = self.youtube_block.get_transcript("test_video_id", self.credentials)

        # Assert
        assert result == mock_transcript
        mock_api_class.assert_called_once()
        proxy_config = mock_api_class.call_args[1]["proxy_config"]
        assert isinstance(proxy_config, WebshareProxyConfig)
        mock_api.fetch.assert_called_once_with(video_id="test_video_id")
        mock_api.list.assert_not_called()

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_with_custom_credentials(self, mock_api_class):
        """Test getting transcript with custom proxy credentials."""
        # Setup mock
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_transcript = Mock(spec=FetchedTranscript)
        mock_api.fetch.return_value = mock_transcript

        credentials = UserPasswordCredentials(
            provider=ProviderName.WEBSHARE_PROXY,
            username=SecretStr("custom_user"),
            password=SecretStr("custom_pass"),
        )

        # Execute
        result = self.youtube_block.get_transcript("test_video_id", credentials)

        # Assert
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
        # Setup mock
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Create mock transcript list with Hungarian transcript
        mock_transcript_list = Mock()
        mock_transcript_hu = Mock(spec=Transcript)
        mock_fetched_transcript = Mock(spec=FetchedTranscript)
        mock_transcript_hu.fetch.return_value = mock_fetched_transcript

        # Set up the transcript list to have manually created transcripts empty
        # and generated transcripts with Hungarian
        mock_transcript_list._manually_created_transcripts = {}
        mock_transcript_list._generated_transcripts = {"hu": mock_transcript_hu}

        # Mock API to raise NoTranscriptFound for English, then return list
        mock_api.fetch.side_effect = NoTranscriptFound(
            "test_video_id", ("en",), mock_transcript_list
        )
        mock_api.list.return_value = mock_transcript_list

        # Execute
        result = self.youtube_block.get_transcript("test_video_id", self.credentials)

        # Assert
        assert result == mock_fetched_transcript
        mock_api_class.assert_called_once()
        mock_api.fetch.assert_called_once_with(video_id="test_video_id")
        mock_api.list.assert_called_once_with("test_video_id")
        mock_transcript_hu.fetch.assert_called_once()

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_prefers_manually_created(self, mock_api_class):
        """Test that manually created transcripts are preferred over generated ones."""
        # Setup mock
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Create mock transcript list with both manual and generated transcripts
        mock_transcript_list = Mock()
        mock_transcript_manual = Mock(spec=Transcript)
        mock_transcript_generated = Mock(spec=Transcript)
        mock_fetched_manual = Mock(spec=FetchedTranscript)
        mock_transcript_manual.fetch.return_value = mock_fetched_manual

        # Set up the transcript list
        mock_transcript_list._manually_created_transcripts = {
            "es": mock_transcript_manual
        }
        mock_transcript_list._generated_transcripts = {"hu": mock_transcript_generated}

        # Mock API to raise NoTranscriptFound for English
        mock_api.fetch.side_effect = NoTranscriptFound(
            "test_video_id", ("en",), mock_transcript_list
        )
        mock_api.list.return_value = mock_transcript_list

        # Execute
        result = self.youtube_block.get_transcript("test_video_id", self.credentials)

        # Assert - should use manually created transcript first
        assert result == mock_fetched_manual
        mock_api_class.assert_called_once()
        mock_transcript_manual.fetch.assert_called_once()
        mock_transcript_generated.fetch.assert_not_called()

    @patch("backend.blocks.youtube.YouTubeTranscriptApi")
    def test_get_transcript_no_transcripts_available(self, mock_api_class):
        """Test that exception is re-raised when no transcripts are available at all."""
        # Setup mock
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Create mock transcript list with no transcripts
        mock_transcript_list = Mock()
        mock_transcript_list._manually_created_transcripts = {}
        mock_transcript_list._generated_transcripts = {}

        # Mock API to raise NoTranscriptFound
        original_exception = NoTranscriptFound(
            "test_video_id", ("en",), mock_transcript_list
        )
        mock_api.fetch.side_effect = original_exception
        mock_api.list.return_value = mock_transcript_list

        # Execute and assert exception is raised
        with pytest.raises(NoTranscriptFound):
            self.youtube_block.get_transcript("test_video_id", self.credentials)
        mock_api_class.assert_called_once()
