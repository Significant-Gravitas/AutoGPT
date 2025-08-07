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

    # Calendar Management Endpoints

    async def create_calendar(
        self,
        oauth_client_id: str,
        oauth_client_secret: str,
        oauth_refresh_token: str,
        platform: str,
        raw_calendar_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Connect a new calendar integration.

        POST /calendars
        """
        body = {
            "oauth_client_id": oauth_client_id,
            "oauth_client_secret": oauth_client_secret,
            "oauth_refresh_token": oauth_refresh_token,
            "platform": platform,
        }

        if raw_calendar_id is not None:
            body["raw_calendar_id"] = raw_calendar_id

        response = await self.requests.post(
            f"{self.BASE_URL}/calendars",
            headers=self.headers,
            json=body,
        )
        return response.json()

    async def list_calendars(self) -> List[Dict[str, Any]]:
        """
        List all integrated calendars.

        GET /calendars
        """
        response = await self.requests.get(
            f"{self.BASE_URL}/calendars",
            headers=self.headers,
        )
        result = response.json()
        # Ensure we return a list
        if isinstance(result, list):
            return result
        return []

    async def update_calendar(
        self,
        calendar_id: str,
        oauth_client_id: str,
        oauth_client_secret: str,
        oauth_refresh_token: str,
        platform: str,
    ) -> Dict[str, Any]:
        """
        Update calendar credentials or platform.

        PATCH /calendars/{uuid}
        """
        body = {
            "oauth_client_id": oauth_client_id,
            "oauth_client_secret": oauth_client_secret,
            "oauth_refresh_token": oauth_refresh_token,
            "platform": platform,
        }

        response = await self.requests.patch(
            f"{self.BASE_URL}/calendars/{calendar_id}",
            headers=self.headers,
            json=body,
        )
        return response.json()

    async def delete_calendar(self, calendar_id: str) -> bool:
        """
        Remove a calendar integration.

        DELETE /calendars/{uuid}
        """
        response = await self.requests.delete(
            f"{self.BASE_URL}/calendars/{calendar_id}",
            headers=self.headers,
        )
        return response.status in [200, 204]

    # Calendar Events Endpoints (not in official API docs but used in current implementation)
    # These endpoints might be undocumented or from a different API version

    async def list_calendar_events(
        self,
        calendar_id: str,
        start_date_gte: Optional[str] = None,
        start_date_lte: Optional[str] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List events for a specific calendar.

        GET /calendar_events (possibly undocumented)
        """
        params = {"calendar_id": calendar_id}

        if start_date_gte:
            params["start_date_gte"] = start_date_gte
        if start_date_lte:
            params["start_date_lte"] = start_date_lte
        if cursor:
            params["cursor"] = cursor

        response = await self.requests.get(
            f"{self.BASE_URL}/calendar_events",
            headers=self.headers,
            params=params,
        )
        return response.json()

    async def get_calendar_event(self, event_id: str) -> Dict[str, Any]:
        """
        Get details for a specific calendar event.

        GET /calendar_events/{event_id} (possibly undocumented)
        """
        response = await self.requests.get(
            f"{self.BASE_URL}/calendar_events/{event_id}",
            headers=self.headers,
        )
        return response.json()

    async def schedule_bot_for_event(
        self,
        event_id: str,
        bot_config: Dict[str, Any],
        all_occurrences: bool = False,
    ) -> Dict[str, Any]:
        """
        Schedule a bot for a calendar event.

        POST /calendar_events/{event_id}/bot (possibly undocumented)
        """
        params = {"all_occurrences": str(all_occurrences).lower()}

        response = await self.requests.post(
            f"{self.BASE_URL}/calendar_events/{event_id}/bot",
            headers=self.headers,
            params=params,
            json=bot_config,
        )
        return response.json()

    async def unschedule_bot_from_event(
        self, event_id: str, all_occurrences: bool = False
    ) -> Dict[str, Any]:
        """
        Remove a scheduled bot from an event.

        DELETE /calendar_events/{event_id}/bot (possibly undocumented)
        """
        params = {"all_occurrences": str(all_occurrences).lower()}

        response = await self.requests.delete(
            f"{self.BASE_URL}/calendar_events/{event_id}/bot",
            headers=self.headers,
            params=params,
        )
        return response.json()

    async def patch_bot_for_event(
        self,
        event_id: str,
        bot_patch: Dict[str, Any],
        all_occurrences: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Update bot configuration for an event.

        PATCH /calendar_events/{event_id}/bot (possibly undocumented)
        """
        params = {}
        if all_occurrences is not None:
            params["all_occurrences"] = str(all_occurrences).lower()

        response = await self.requests.patch(
            f"{self.BASE_URL}/calendar_events/{event_id}/bot",
            headers=self.headers,
            params=params,
            json=bot_patch,
        )
        return response.json()

    # Internal/Maintenance Endpoints (possibly undocumented)

    async def resync_all_calendars(self) -> Dict[str, Any]:
        """
        Force re-sync of all calendars.

        POST /internal/calendar/resync_all (possibly internal endpoint)
        """
        response = await self.requests.post(
            f"{self.BASE_URL}/internal/calendar/resync_all",
            headers=self.headers,
        )
        return response.json()
