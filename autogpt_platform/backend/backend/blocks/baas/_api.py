"""
Meeting BaaS API client module.
All API calls centralized for consistency and maintainability.
"""

from typing import Any, Dict, List, Optional

from backend.sdk import Requests


class MeetingBaasAPI:
    """Client for Meeting BaaS API endpoints."""

    BASE_URL = "https://api.meetingbaas.com"

    def __init__(self, api_key: str):
        """Initialize API client with authentication key."""
        self.api_key = api_key
        self.headers = {"x-meeting-baas-api-key": api_key}
        self.requests = Requests()

    # Bot Management Endpoints

    async def join_meeting(
        self,
        bot_name: str,
        meeting_url: str,
        reserved: bool = False,
        bot_image: Optional[str] = None,
        entry_message: Optional[str] = None,
        start_time: Optional[int] = None,
        speech_to_text: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
        automatic_leave: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        recording_mode: str = "speaker_view",
        streaming: Optional[Dict[str, Any]] = None,
        deduplication_key: Optional[str] = None,
        zoom_sdk_id: Optional[str] = None,
        zoom_sdk_pwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deploy a bot to join and record a meeting.

        POST /bots
        """
        body = {
            "bot_name": bot_name,
            "meeting_url": meeting_url,
            "reserved": reserved,
            "recording_mode": recording_mode,
        }

        # Add optional fields if provided
        if bot_image is not None:
            body["bot_image"] = bot_image
        if entry_message is not None:
            body["entry_message"] = entry_message
        if start_time is not None:
            body["start_time"] = start_time
        if speech_to_text is not None:
            body["speech_to_text"] = speech_to_text
        if webhook_url is not None:
            body["webhook_url"] = webhook_url
        if automatic_leave is not None:
            body["automatic_leave"] = automatic_leave
        if extra is not None:
            body["extra"] = extra
        if streaming is not None:
            body["streaming"] = streaming
        if deduplication_key is not None:
            body["deduplication_key"] = deduplication_key
        if zoom_sdk_id is not None:
            body["zoom_sdk_id"] = zoom_sdk_id
        if zoom_sdk_pwd is not None:
            body["zoom_sdk_pwd"] = zoom_sdk_pwd

        response = await self.requests.post(
            f"{self.BASE_URL}/bots",
            headers=self.headers,
            json=body,
        )
        return response.json()

    async def leave_meeting(self, bot_id: str) -> bool:
        """
        Remove a bot from an ongoing meeting.

        DELETE /bots/{uuid}
        """
        response = await self.requests.delete(
            f"{self.BASE_URL}/bots/{bot_id}",
            headers=self.headers,
        )
        return response.status in [200, 204]

    async def retranscribe(
        self,
        bot_uuid: str,
        speech_to_text: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Re-run transcription on a bot's audio.

        POST /bots/retranscribe
        """
        body: Dict[str, Any] = {"bot_uuid": bot_uuid}

        if speech_to_text is not None:
            body["speech_to_text"] = speech_to_text
        if webhook_url is not None:
            body["webhook_url"] = webhook_url

        response = await self.requests.post(
            f"{self.BASE_URL}/bots/retranscribe",
            headers=self.headers,
            json=body,
        )

        if response.status == 202:
            return {"accepted": True}
        return response.json()

    # Data Retrieval Endpoints

    async def get_meeting_data(
        self, bot_id: str, include_transcripts: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve meeting data including recording and transcripts.

        GET /bots/meeting_data
        """
        params = {
            "bot_id": bot_id,
            "include_transcripts": str(include_transcripts).lower(),
        }

        response = await self.requests.get(
            f"{self.BASE_URL}/bots/meeting_data",
            headers=self.headers,
            params=params,
        )
        return response.json()

    async def get_screenshots(self, bot_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve screenshots captured during a meeting.

        GET /bots/{uuid}/screenshots
        """
        response = await self.requests.get(
            f"{self.BASE_URL}/bots/{bot_id}/screenshots",
            headers=self.headers,
        )
        result = response.json()
        # Ensure we return a list
        if isinstance(result, list):
            return result
        return []

    async def delete_data(self, bot_id: str) -> bool:
        """
        Delete a bot's recorded data.

        POST /bots/{uuid}/delete_data
        """
        response = await self.requests.post(
            f"{self.BASE_URL}/bots/{bot_id}/delete_data",
            headers=self.headers,
        )
        return response.status == 200

    async def list_bots_with_metadata(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        filter_by: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        List bots with metadata including IDs, names, and meeting details.

        GET /bots/bots_with_metadata
        """
        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if sort_by is not None:
            params["sort_by"] = sort_by
        if sort_order is not None:
            params["sort_order"] = sort_order
        if filter_by is not None:
            params.update(filter_by)

        response = await self.requests.get(
            f"{self.BASE_URL}/bots/bots_with_metadata",
            headers=self.headers,
            params=params,
        )
        return response.json()
