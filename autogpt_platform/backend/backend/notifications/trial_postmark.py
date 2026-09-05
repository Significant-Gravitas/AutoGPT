"""Trial-only Postmark delivery and reconciliation by durable notice identity."""

import httpx
from prisma.enums import NotificationType
from pydantic import BaseModel, Field

from backend.data.notifications import TrialUpdateData
from backend.notifications.renderer import build_urls, render
from backend.util.settings import Settings


class AcceptedMessage(BaseModel):
    message_id: str = Field(alias="MessageID", min_length=1)
    error_code: int = Field(alias="ErrorCode")


class OutboundMessage(BaseModel):
    message_id: str = Field(alias="MessageID", min_length=1)
    metadata: dict[str, str] = Field(alias="Metadata")


class OutboundSearch(BaseModel):
    messages: list[OutboundMessage] = Field(alias="Messages")


class TrialEmailSender:
    async def send(self, delivery_id: str, email: str, data: TrialUpdateData) -> str:
        settings = Settings()
        rendered = render(NotificationType.TRIAL_UPDATE, data, email, build_urls(""))
        async with self._client() as client:
            response = await client.post(
                "/email",
                json={
                    "From": settings.config.billing_sender_email,
                    "To": email,
                    "Subject": rendered.subject,
                    "HtmlBody": rendered.html,
                    "TextBody": rendered.text,
                    "MessageStream": settings.config.postmark_transactional_stream,
                    "Metadata": {"trial_notice_id": delivery_id},
                },
            )
        response.raise_for_status()
        result = AcceptedMessage.model_validate(response.json())
        if result.error_code:
            raise RuntimeError("Postmark refused the trial notice")
        return result.message_id

    async def find_accepted(self, delivery_id: str) -> str | None:
        async with self._client() as client:
            response = await client.get(
                "/messages/outbound",
                params={
                    "count": "2",
                    "offset": "0",
                    "metadata_trial_notice_id": delivery_id,
                    "messagestream": Settings().config.postmark_transactional_stream,
                },
            )
        response.raise_for_status()
        results = OutboundSearch.model_validate(response.json())
        for message in results.messages:
            if message.metadata.get("trial_notice_id") != delivery_id:
                raise ValueError("Postmark search returned another trial notice")
        return results.messages[0].message_id if results.messages else None

    def _client(self) -> httpx.AsyncClient:
        token = Settings().secrets.postmark_server_api_token
        if not token:
            raise RuntimeError("Postmark is not configured for trial notices")
        return httpx.AsyncClient(
            base_url="https://api.postmarkapp.com",
            timeout=30,
            headers={"X-Postmark-Server-Token": token, "Accept": "application/json"},
        )
