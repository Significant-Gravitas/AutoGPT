from datetime import datetime, timedelta

from backend.blocks.hubspot._auth import (
    HubSpotCredentials,
    HubSpotCredentialsField,
    HubSpotCredentialsInput,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.request import requests


class HubSpotEngagementBlock(Block):
    class Input(BlockSchema):
        credentials: HubSpotCredentialsInput = HubSpotCredentialsField()
        operation: str = SchemaField(
            description="Operation to perform (send_email, track_engagement)",
            default="send_email",
        )
        email_data: dict = SchemaField(
            description="Email data including recipient, subject, content",
            default={},
        )
        contact_id: str = SchemaField(
            description="Contact ID for engagement tracking", default=""
        )
        timeframe_days: int = SchemaField(
            description="Number of days to look back for engagement",
            default=30,
            optional=True,
        )

    class Output(BlockSchema):
        result: dict = SchemaField(description="Operation result")
        status: str = SchemaField(description="Operation status")

    def __init__(self):
        super().__init__(
            id="c6524385-7d87-49d6-a470-248bd29ca765",
            description="Manages HubSpot engagements - sends emails and tracks engagement metrics",
            categories={BlockCategory.CRM, BlockCategory.COMMUNICATION},
            input_schema=HubSpotEngagementBlock.Input,
            output_schema=HubSpotEngagementBlock.Output,
        )

    def run(
        self, input_data: Input, *, credentials: HubSpotCredentials, **kwargs
    ) -> BlockOutput:
        base_url = "https://api.hubapi.com"
        headers = {
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        if input_data.operation == "send_email":
            # Using the email send API
            email_url = f"{base_url}/crm/v3/objects/emails"
            email_data = {
                "properties": {
                    "hs_timestamp": datetime.now().isoformat(),
                    "hubspot_owner_id": "1",  # This should be configurable
                    "hs_email_direction": "OUTBOUND",
                    "hs_email_status": "SEND",
                    "hs_email_subject": input_data.email_data.get("subject"),
                    "hs_email_text": input_data.email_data.get("content"),
                    "hs_email_to_email": input_data.email_data.get("recipient"),
                }
            }

            response = requests.post(email_url, headers=headers, json=email_data)
            result = response.json()
            yield "result", result
            yield "status", "email_sent"

        elif input_data.operation == "track_engagement":
            # Get engagement events for the contact
            from_date = datetime.now() - timedelta(days=input_data.timeframe_days)
            engagement_url = (
                f"{base_url}/crm/v3/objects/contacts/{input_data.contact_id}/engagement"
            )

            params = {"limit": 100, "after": from_date.isoformat()}

            response = requests.get(engagement_url, headers=headers, params=params)
            engagements = response.json()

            # Process engagement metrics
            metrics = {
                "email_opens": 0,
                "email_clicks": 0,
                "email_replies": 0,
                "last_engagement": None,
                "engagement_score": 0,
            }

            for engagement in engagements.get("results", []):
                eng_type = engagement.get("properties", {}).get("hs_engagement_type")
                if eng_type == "EMAIL":
                    metrics["email_opens"] += 1
                elif eng_type == "EMAIL_CLICK":
                    metrics["email_clicks"] += 1
                elif eng_type == "EMAIL_REPLY":
                    metrics["email_replies"] += 1

                # Update last engagement time
                eng_time = engagement.get("properties", {}).get("hs_timestamp")
                if eng_time and (
                    not metrics["last_engagement"]
                    or eng_time > metrics["last_engagement"]
                ):
                    metrics["last_engagement"] = eng_time

            # Calculate simple engagement score
            metrics["engagement_score"] = (
                metrics["email_opens"]
                + metrics["email_clicks"] * 2
                + metrics["email_replies"] * 3
            )

            yield "result", metrics
            yield "status", "engagement_tracked"
