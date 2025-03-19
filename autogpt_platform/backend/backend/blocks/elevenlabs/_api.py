"""
API module for ElevenLabs API integration.

This module provides a client for the ElevenLabs API, which offers text-to-speech
and speech-to-speech capabilities.
"""

import json
from io import BytesIO
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from backend.data.model import APIKeyCredentials
from backend.util.request import Requests


class ElevenLabsException(Exception):
    """Exception raised for ElevenLabs API errors."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class VoiceSettings(BaseModel):
    """Model for voice settings."""

    stability: float
    similarity_boost: float


class ElevenLabsClient:
    """Client for the ElevenLabs API."""

    API_BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(
        self,
        credentials: Optional[APIKeyCredentials] = None,
        custom_requests: Optional[Requests] = None,
    ):
        """
        Initialize the ElevenLabs API client.

        Args:
            credentials: API key credentials for ElevenLabs.
            custom_requests: Custom request handler (for testing).
        """
        if custom_requests:
            self._requests = custom_requests
        else:
            headers: Dict[str, str] = {
                "Content-Type": "application/json",
            }
            if credentials:
                headers["xi-api-key"] = credentials.api_key.get_secret_value()

            self._requests = Requests(
                extra_headers=headers,
                raise_for_status=False,
            )

    def _handle_response(self, response, raw_binary=False) -> Any:
        """
        Handle API response and check for errors.

        Args:
            response: Response object from the request.
            raw_binary: If True, return the raw binary content instead of parsing as JSON.

        Returns:
            Parsed response data.

        Raises:
            ElevenLabsException: If the API request fails.
        """
        if not response.ok:
            error_message = "Unknown error"
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_message = error_data.get(
                        "detail", error_data.get("message", "Unknown error")
                    )
            except Exception:
                error_message = response.text or f"Error: HTTP {response.status_code}"

            raise ElevenLabsException(
                f"ElevenLabs API error ({response.status_code}): {error_message}",
                response.status_code,
            )

        if raw_binary:
            return response.content

        try:
            return response.json()
        except Exception:
            return response.content

    # === Voice Management Endpoints ===

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Retrieve a list of all available TTS and STS models.

        Returns:
            List of model objects.
        """
        response = self._requests.get(f"{self.API_BASE_URL}/models")
        return self._handle_response(response)

    def get_voices(self) -> Dict[str, Any]:
        """
        Retrieve a list of all voices available to the user.

        Returns:
            Dictionary containing the list of voices.
        """
        response = self._requests.get(f"{self.API_BASE_URL}/voices")
        return self._handle_response(response)

    def get_voice(self, voice_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata about a specific voice.

        Args:
            voice_id: ID of the voice to retrieve.

        Returns:
            Voice metadata.
        """
        response = self._requests.get(f"{self.API_BASE_URL}/voices/{voice_id}")
        return self._handle_response(response)

    def add_voice(
        self,
        name: str,
        files: List[BytesIO],
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Add a new voice by uploading audio samples.

        Args:
            name: Name for the new voice.
            files: List of audio file objects.
            description: Optional description for the voice.
            labels: Optional labels as key-value pairs.

        Returns:
            Dictionary with the new voice_id.
        """
        data = {"name": name}
        if description:
            data["description"] = description
        if labels:
            data["labels"] = json.dumps(labels)

        files_data = []
        for i, file_obj in enumerate(files):
            files_data.append(("files", (f"sample_{i}.mp3", file_obj, "audio/mpeg")))

        # Custom request for multipart/form-data
        response = self._requests.post(
            f"{self.API_BASE_URL}/voices/add",
            headers={"xi-api-key": self._requests.extra_headers.get("xi-api-key", "") if self._requests.extra_headers else ""},
            data=data,
            files=files_data,
        )
        return self._handle_response(response)

    def edit_voice(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        files: Optional[List[BytesIO]] = None,
    ) -> None:
        """
        Edit an existing voice.

        Args:
            voice_id: ID of the voice to edit.
            name: Optional new name for the voice.
            description: Optional new description.
            labels: Optional new labels.
            files: Optional new audio samples.
        """
        data = {}
        if name:
            data["name"] = name
        if description:
            data["description"] = description
        if labels:
            data["labels"] = json.dumps(labels)

        files_data = []
        if files:
            for i, file_obj in enumerate(files):
                files_data.append(
                    ("files", (f"sample_{i}.mp3", file_obj, "audio/mpeg"))
                )

        # Custom request for multipart/form-data
        response = self._requests.post(
            f"{self.API_BASE_URL}/voices/{voice_id}/edit",
            headers={"xi-api-key": self._requests.extra_headers.get("xi-api-key", "") if self._requests.extra_headers else ""},
            data=data,
            files=files_data,
        )
        self._handle_response(response)

    def delete_voice(self, voice_id: str) -> None:
        """
        Delete a voice.

        Args:
            voice_id: ID of the voice to delete.
        """
        response = self._requests.delete(f"{self.API_BASE_URL}/voices/{voice_id}")
        self._handle_response(response)

    def get_voice_settings(self, voice_id: str) -> Dict[str, float]:
        """
        Retrieve a voice's settings.

        Args:
            voice_id: ID of the voice.

        Returns:
            Voice settings (stability and similarity_boost).
        """
        response = self._requests.get(f"{self.API_BASE_URL}/voices/{voice_id}/settings")
        return self._handle_response(response)

    def edit_voice_settings(
        self, voice_id: str, stability: float, similarity_boost: float
    ) -> None:
        """
        Edit a voice's settings.

        Args:
            voice_id: ID of the voice.
            stability: Stability setting (0.0 to 1.0).
            similarity_boost: Similarity boost setting (0.0 to 1.0).
        """
        data = {"stability": stability, "similarity_boost": similarity_boost}
        response = self._requests.post(
            f"{self.API_BASE_URL}/voices/{voice_id}/settings/edit",
            json=data,
        )
        self._handle_response(response)

    # === Text-to-Speech Endpoints ===

    def text_to_speech(
        self,
        voice_id: str,
        text: str,
        model_id: Optional[str] = None,
        voice_settings: Optional[VoiceSettings] = None,
    ) -> bytes:
        """
        Convert text to speech using a specific voice.

        Args:
            voice_id: ID of the voice to use.
            text: Text to convert to speech.
            model_id: Optional model ID to use (e.g., "eleven_multilingual_v2").
            voice_settings: Optional voice settings to override defaults.

        Returns:
            Audio data as bytes.
        """
        data = {"text": text }
        if model_id:
            data["model_id"] = model_id
        if voice_settings:
            data["voice_settings"] = voice_settings.model_dump()
           
        response = self._requests.post(
            f"{self.API_BASE_URL}/text-to-speech/{voice_id}",
            json=data,
        )
        return self._handle_response(response, raw_binary=True)

    def text_to_speech_stream(
        self,
        voice_id: str,
        text: str,
        model_id: Optional[str] = None,
        voice_settings: Optional[VoiceSettings] = None,
    ) -> bytes:
        """
        Stream text-to-speech in real-time.

        Args:
            voice_id: ID of the voice to use.
            text: Text to convert to speech.
            model_id: Optional model ID to use.
            voice_settings: Optional voice settings.

        Returns:
            Complete audio data as bytes after stream finishes.
        """
        data = {"text": text}
        if model_id:
            data["model_id"] = model_id
        if voice_settings:
            data["voice_settings"] = json.dumps(voice_settings.dict())

        response = self._requests.post(
            f"{self.API_BASE_URL}/text-to-speech/{voice_id}/stream",
            json=data,
        )
        return self._handle_response(response, raw_binary=True)

    # === Speech-to-Speech Endpoints ===

    def speech_to_speech(
        self,
        voice_id: str,
        audio: BytesIO,
        voice_settings: Optional[VoiceSettings] = None,
    ) -> bytes:
        """
        Transform audio from one voice to another.

        Args:
            voice_id: ID of the target voice.
            audio: Input audio file.
            voice_settings: Optional voice settings.

        Returns:
            Transformed audio data as bytes.
        """
        data = {}
        if voice_settings:
            data["voice_settings"] = json.dumps(voice_settings.dict())

        files = [("audio", ("input.mp3", audio, "audio/mpeg"))]

        # Custom request for multipart/form-data
        response = self._requests.post(
            f"{self.API_BASE_URL}/speech-to-speech/{voice_id}",
            headers={"xi-api-key": self._requests.extra_headers.get("xi-api-key", "") if self._requests.extra_headers else ""},
            data=data,
            files=files,
        )
        return self._handle_response(response, raw_binary=True)

    def speech_to_speech_stream(
        self,
        voice_id: str,
        audio: BytesIO,
        voice_settings: Optional[VoiceSettings] = None,
    ) -> bytes:
        """
        Stream speech-to-speech transformation in real-time.

        Args:
            voice_id: ID of the target voice.
            audio: Input audio file.
            voice_settings: Optional voice settings.

        Returns:
            Complete audio data as bytes after stream finishes.
        """
        data = {}
        if voice_settings:
            data["voice_settings"] = json.dumps(voice_settings.dict())

        files = [("audio", ("input.mp3", audio, "audio/mpeg"))]

        # Custom request for multipart/form-data
        response = self._requests.post(
            f"{self.API_BASE_URL}/speech-to-speech/{voice_id}/stream",
            headers={"xi-api-key": self._requests.extra_headers.get("xi-api-key", "") if self._requests.extra_headers else ""},
            data=data,
            files=files,
        )
        return self._handle_response(response, raw_binary=True)

    # === History Endpoints ===

    def get_history(
        self,
        page_size: Optional[int] = None,
        start_after_history_item_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve a list of generation history items.

        Args:
            page_size: Optional number of items per page.
            start_after_history_item_id: Optional pagination marker.

        Returns:
            History data.
        """
        params = {}
        if page_size:
            params["page_size"] = page_size
        if start_after_history_item_id:
            params["start_after_history_item_id"] = start_after_history_item_id

        response = self._requests.get(f"{self.API_BASE_URL}/history", params=params)
        return self._handle_response(response)

    def get_history_item(self, history_item_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a specific history item.

        Args:
            history_item_id: ID of the history item.

        Returns:
            History item metadata.
        """
        response = self._requests.get(f"{self.API_BASE_URL}/history/{history_item_id}")
        return self._handle_response(response)

    def get_history_audio(self, history_item_id: str) -> bytes:
        """
        Download audio from a history item.

        Args:
            history_item_id: ID of the history item.

        Returns:
            Audio data as bytes.
        """
        response = self._requests.get(
            f"{self.API_BASE_URL}/history/{history_item_id}/audio"
        )
        return self._handle_response(response, raw_binary=True)

    def delete_history_item(self, history_item_id: str) -> None:
        """
        Delete a history item.

        Args:
            history_item_id: ID of the history item to delete.
        """
        response = self._requests.delete(
            f"{self.API_BASE_URL}/history/{history_item_id}"
        )
        self._handle_response(response)

    def download_history_items(self, history_item_ids: List[str]) -> bytes:
        """
        Download multiple history items as a ZIP file.

        Args:
            history_item_ids: List of history item IDs to download.

        Returns:
            ZIP file data as bytes.
        """
        data = {"history_item_ids": history_item_ids}
        response = self._requests.post(
            f"{self.API_BASE_URL}/history/download",
            json=data,
        )
        return self._handle_response(response, raw_binary=True)

    # === User Endpoints ===

    def get_user_info(self) -> Dict[str, Any]:
        """
        Retrieve user account information.

        Returns:
            User information.
        """
        response = self._requests.get(f"{self.API_BASE_URL}/user")
        return self._handle_response(response)

    def get_subscription_info(self) -> Dict[str, Any]:
        """
        Retrieve subscription details.

        Returns:
            Subscription information.
        """
        response = self._requests.get(f"{self.API_BASE_URL}/user/subscription")
        return self._handle_response(response)
