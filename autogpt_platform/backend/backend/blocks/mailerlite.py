import json
from typing import Dict, List, Literal

import mailerlite
from autogpt_libs.supabase_integration_credentials_store.types import APIKeyCredentials
from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsField, CredentialsMetaInput, SchemaField
import logging

logger = logging.getLogger(__name__)

# Test credentials for both blocks
TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="mailerlite",
    api_key=SecretStr("mock-mailerlite-api-key"),
    title="Mock MailerLite API key",
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


class MailerLiteSubscribeBlock(Block):
    """Block for adding new subscribers to MailerLite."""
    
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[Literal["mailerlite"], Literal["api_key"]] = CredentialsField(
            provider="mailerlite",
            supported_credential_types={"api_key"},
            description="Enter your MailerLite API key. You can find or create your API key in your MailerLite account under Integrations > API Keys.",
        )
        email: str = SchemaField(
            description="Email address of the new subscriber",
            placeholder="e.g., user@example.com",
        )
        first_name: str = SchemaField(
            description="First name of the subscriber",
            placeholder="e.g., John",
            default="",
        )
        last_name: str = SchemaField(
            description="Last name of the subscriber",
            placeholder="e.g., Doe",
            default="",
        )
        ip_address: str = SchemaField(
            description="IP address of the subscriber for compliance (optional)",
            placeholder="e.g., 192.168.1.1",
            default="",
        )
        group_ids: List[str] = SchemaField(
            description="List of group IDs to add the subscriber to",
            placeholder="e.g., ['1234567', '7654321']",
            default=[],
        )

    class Output(BlockSchema):
        subscriber_id: int = SchemaField(description="ID of the created/updated subscriber")
        subscriber_email: str = SchemaField(description="Email of the created/updated subscriber")
        subscriber_status: str = SchemaField(description="Status of the subscriber (active, unsubscribed, etc.)")
        subscriber_source: str = SchemaField(description="Source of the subscriber")
        subscriber_stats: str = SchemaField(description="Subscriber engagement statistics as JSON string")
        signup_date: str = SchemaField(description="When the subscriber signed up")
        signup_ip: str = SchemaField(description="IP address used for signup")
        subscriber_fields: str = SchemaField(description="Additional fields for the subscriber as JSON string")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="83222e0e-c599-4c8a-b7e0-ab8f2463b573",
            description="Add a new subscriber to your MailerLite mailing list",
            categories={BlockCategory.COMMUNICATION},
            input_schema=MailerLiteSubscribeBlock.Input,
            output_schema=MailerLiteSubscribeBlock.Output,
            test_input={
                "credentials": {
                    "provider": "mailerlite",
                    "id": "01234567-89ab-cdef-0123-456789abcdef",
                    "type": "api_key",
                    "title": "Mock MailerLite API key",
                },
                "email": "test@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "ip_address": "192.168.1.1",
                "group_ids": ["1234567"],
            },
            test_output=[
                ("subscriber_id", "31897397363737859"),
                ("subscriber_email", "test@example.com"),
                ("subscriber_status", "active"),
                ("subscriber_source", "api"),
                ("subscriber_stats", '{"sent":0,"opens_count":0,"clicks_count":0,"open_rate":0.0,"click_rate":0.0}'),
                ("signup_date", "2021-08-31 14:22:08"),
                ("signup_ip", "192.168.1.1"),
                ("subscriber_fields", '{"name":"John","last_name":"Doe"}'),
            ],
            test_mock={
                "_create_subscriber": lambda *args, **kwargs: {
                    "data": {
                        "id": "31897397363737859",
                        "email": "test@example.com",
                        "status": "active",
                        "source": "api",
                        "sent": 0,
                        "opens_count": 0,
                        "clicks_count": 0,
                        "open_rate": 0,
                        "click_rate": 0,
                        "ip_address": "192.168.1.1",
                        "subscribed_at": "2021-08-31 14:22:08",
                        "fields": {
                            "name": "John",
                            "last_name": "Doe",
                        },
                    }
                }
            },
        )

    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs) -> BlockOutput:
        try:
            client = mailerlite.Client({
                'api_key': credentials.api_key.get_secret_value()
            })
            
            logger.info(f"[MailerLiteSubscribeBlock] Creating subscriber with email: {input_data.email}")
            response = self._create_subscriber(client, input_data)
            logger.info(f"[MailerLiteSubscribeBlock] Raw API response: {response}")
            
            if isinstance(response, dict):
                if 'data' in response:
                    subscriber_data = response['data']
                else:
                    subscriber_data = response  # Some API endpoints return data directly
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
            # Convert dictionaries to JSON strings
            stats = json.dumps({
                "sent": int(subscriber_data.get("sent", 0)),
                "opens_count": int(subscriber_data.get("opens_count", 0)),
                "clicks_count": int(subscriber_data.get("clicks_count", 0)),
                "open_rate": float(subscriber_data.get("open_rate", 0)),
                "click_rate": float(subscriber_data.get("click_rate", 0))
            })
            
            fields = json.dumps(
                {k: str(v) if v is not None else None 
                 for k, v in subscriber_data.get("fields", {}).items()}
            )
            
            yield "subscriber_id", int(subscriber_data.get("id", "-1"))
            yield "subscriber_email", str(subscriber_data.get("email", ""))
            yield "subscriber_status", str(subscriber_data.get("status", ""))
            yield "subscriber_source", str(subscriber_data.get("source", ""))
            yield "subscriber_stats", stats
            yield "signup_date", str(subscriber_data.get("subscribed_at", ""))
            yield "signup_ip", str(subscriber_data.get("ip_address", ""))
            yield "subscriber_fields", fields
    
        except Exception as e:
            logger.error(f"[MailerLiteSubscribeBlock] API Error: {str(e)}")
            yield "error", str(e)

    def _create_subscriber(self, client: mailerlite.Client, input_data: Input) -> Dict:
        # Build fields dict if name data is provided
        fields = {}
        if input_data.first_name:
            fields["name"] = input_data.first_name
        if input_data.last_name:
            fields["last_name"] = input_data.last_name

        # Build kwargs for optional parameters
        kwargs = {}
        if fields:
            kwargs["fields"] = fields
        if input_data.group_ids:
            kwargs["groups"] = input_data.group_ids
        if input_data.ip_address:
            kwargs["ip_address"] = input_data.ip_address
            kwargs["optin_ip"] = input_data.ip_address

        # Call create with email as positional arg and rest as kwargs
        return client.subscribers.create(input_data.email, **kwargs)
    
class MailerLiteCreateGroupBlock(Block):
    """Block for creating a new subscriber group in MailerLite."""
    
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[Literal["mailerlite"], Literal["api_key"]] = CredentialsField(
            provider="mailerlite",
            supported_credential_types={"api_key"},
            description="Enter your MailerLite API key. You can find or create your API key in your MailerLite account under Integrations > API Keys.",
        )
        group_name: str = SchemaField(
            description="Name of the new subscriber group",
            placeholder="e.g., Newsletter Subscribers 2024",
            max_length=255,
        )

    class Output(BlockSchema):
        group_id: int = SchemaField(description="ID of the created group")
        created_group_name: str = SchemaField(description="Name of the created group")
        active_count: str = SchemaField(description="Number of active subscribers in the group")
        open_rate: str = SchemaField(description="Email open rate for this group")
        click_rate: str = SchemaField(description="Email click rate for this group")
        creation_date: str = SchemaField(description="When the group was created")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="ecea735a-51fa-455c-9ebd-95bf3e1c85a8",
            description="Create a new subscriber group in MailerLite",
            categories={BlockCategory.COMMUNICATION},
            input_schema=MailerLiteCreateGroupBlock.Input,
            output_schema=MailerLiteCreateGroupBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "group_name": "Test Group",
            },
            test_output=[
                (
                    "group_id",
                    "1",
                ),
                (
                    "created_group_name",
                    "Test Group",
                ),
                (
                    "active_count",
                    0,
                ),
                (
                    "open_rate",
                    0.0,
                ),
                (
                    "click_rate",
                    0.0,
                ),
                (
                    "creation_date",
                    "2022-05-25 14:22:44",
                ),
            ],
            test_mock={
                "_create_group": lambda *args, **kwargs: {
                    "data": {
                        "id": "1",
                        "name": "Test Group",
                        "active_count": 0,
                        "open_rate": {"float": 0, "string": "0%"},
                        "click_rate": {"float": 0, "string": "0%"},
                        "created_at": "2022-05-25 14:22:44"
                    }
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs) -> BlockOutput:
        try:
            client = mailerlite.Client({
                'api_key': credentials.api_key.get_secret_value()
            })
            
            try:
                response = self._create_group(client, input_data.group_name)
                logger.info(f"[MailerLiteCreateGroupBlock] API Response: {response}")
                
                # Handle both possible response formats
                if isinstance(response, dict):
                    if 'data' in response:
                        group_data = response['data']
                    else:
                        group_data = response  # Some API endpoints return data directly
                else:
                    raise ValueError(f"Unexpected response type: {type(response)}")
                
                yield "group_id", int(group_data.get("id", "-1"))
                yield "created_group_name", str(group_data.get("name", ""))
                yield "active_count", str(group_data.get("active_count", 0))
                yield "open_rate", str(
                    group_data.get("open_rate", {}).get("float", 0) 
                    if isinstance(group_data.get("open_rate"), dict) 
                    else 0
                )
                yield "click_rate", str(
                    group_data.get("click_rate", {}).get("float", 0)
                    if isinstance(group_data.get("click_rate"), dict)
                    else 0
                )
                yield "creation_date", str(group_data.get("created_at", ""))

            except Exception as e:
                logger.error(f"[MailerLiteCreateGroupBlock] API Error: {str(e)}")
                raise

        except Exception as e:
            yield "error", str(e)

    def _create_group(self, client: mailerlite.Client, group_name: str) -> Dict:
        logger.info(f"[MailerLiteCreateGroupBlock] Creating group with name: {group_name}")
        response = client.groups.create(group_name)
        logger.info(f"[MailerLiteCreateGroupBlock] Raw API response: {response}")
        logger.info(f"[MailerLiteCreateGroupBlock] Response type: {type(response)}")
        
        return response
    
class MailerLiteAssignToGroupBlock(Block):
    """Block for assigning a subscriber to a group in MailerLite."""
    
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[Literal["mailerlite"], Literal["api_key"]] = CredentialsField(
            provider="mailerlite",
            supported_credential_types={"api_key"},
            description="Enter your MailerLite API key. You can find or create your API key in your MailerLite account under Integrations > API Keys.",
        )
        subscriber_id: int = SchemaField(
            description="ID of the subscriber to assign",
            placeholder="e.g., 111222",
        )
        group_id: int = SchemaField(
            description="ID of the group to assign the subscriber to",
            placeholder="e.g., 1234567",
        )

    class Output(BlockSchema):
        assigned_group_id: int = SchemaField(description="ID of the group")
        assigned_group_name: str = SchemaField(description="Name of the group")
        group_stats: str = SchemaField(description="Statistics about the group as JSON string")
        assignment_date: str = SchemaField(description="When the assignment was made")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="8d77b514-6ce2-4b7d-b3ae-a5afd9e6bee6",
            description="Assign a subscriber to a group in MailerLite",
            categories={BlockCategory.COMMUNICATION},
            input_schema=MailerLiteAssignToGroupBlock.Input,
            output_schema=MailerLiteAssignToGroupBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "subscriber_id": "111222",
                "group_id": "1234567",
            },
            test_output=[
                (
                    "assigned_group_id",
                    "1",
                ),
                (
                    "assigned_group_name",
                    "Test Group",
                ),
                (
                    "group_stats",
                    {
                        "active_count": 0,
                        "sent_count": 0,
                        "opens_count": 0,
                        "open_rate": 0,
                        "clicks_count": 0,
                        "click_rate": 0,
                        "unsubscribed_count": 0,
                        "bounced_count": 0,
                        "junk_count": 0,
                    }
                ),
                (
                    "assignment_date",
                    "2022-05-25 14:22:44",
                ),
            ],
            test_mock={
                "_assign_to_group": lambda *args, **kwargs: {
                    "data": {
                        "id": "1",
                        "name": "Test Group",
                        "active_count": 0,
                        "sent_count": 0,
                        "opens_count": 0,
                        "open_rate": {"float": 0},
                        "clicks_count": 0,
                        "click_rate": {"float": 0},
                        "unsubscribed_count": 0,
                        "bounced_count": 0,
                        "junk_count": 0,
                        "created_at": "2022-05-25 14:22:44"
                    }
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def _assign_to_group(self, client: mailerlite.Client, subscriber_id: int, group_id: int) -> dict:
        return client.subscribers.assign_subscriber_to_group(subscriber_id, group_id)

    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs) -> BlockOutput:
        try:
            client = mailerlite.Client({
                'api_key': credentials.api_key.get_secret_value()
            })
            
            response = self._assign_to_group(client, input_data.subscriber_id, input_data.group_id)
            
            if isinstance(response, dict):
                if 'data' in response:
                    group_data = response['data']
                else:
                    group_data = response  # Some API endpoints return data directly
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
            # Convert stats to JSON string
            stats = json.dumps({
                "active_count": int(group_data.get("active_count", 0)),
                "sent_count": int(group_data.get("sent_count", 0)),
                "opens_count": int(group_data.get("opens_count", 0)),
                "open_rate": float(
                    group_data.get("open_rate", {}).get("float", 0)
                    if isinstance(group_data.get("open_rate"), dict)
                    else 0
                ),
                "clicks_count": int(group_data.get("clicks_count", 0)),
                "click_rate": float(
                    group_data.get("click_rate", {}).get("float", 0)
                    if isinstance(group_data.get("click_rate"), dict)
                    else 0
                ),
                "unsubscribed_count": int(group_data.get("unsubscribed_count", 0)),
                "bounced_count": int(group_data.get("bounced_count", 0)),
                "junk_count": int(group_data.get("junk_count", 0))
            })
            
            yield "assigned_group_id", int(group_data.get("id", "-1"))
            yield "assigned_group_name", str(group_data.get("name", ""))
            yield "group_stats", stats
            yield "assignment_date", str(group_data.get("created_at", ""))

        except Exception as e:
            yield "error", str(e)


class MailerLiteCampaignBlock(Block):
    """Block for creating email campaigns in MailerLite."""
    
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[Literal["mailerlite"], Literal["api_key"]] = CredentialsField(
            provider="mailerlite",
            supported_credential_types={"api_key"},
            description="Enter your MailerLite API key. You can find or create your API key in your MailerLite account under Integrations > API Keys.",
        )
        campaign_name: str = SchemaField(
            description="Name of the email campaign",
            placeholder="e.g., Monthly Newsletter June 2024",
        )
        subject: str = SchemaField(
            description="Email subject line",
            placeholder="e.g., Your Monthly Update is Here!",
        )
        from_name: str = SchemaField(
            description="Sender name that will appear in recipients' inbox",
            placeholder="e.g., John from Company",
        )
        from_email: str = SchemaField(
            description="Sender email address (must be verified in MailerLite)",
            placeholder="e.g., newsletter@company.com",
        )
        content: str = SchemaField(
            description="Email content (supports HTML)",
            placeholder="e.g., Dear subscriber, welcome to our monthly newsletter...",
        )
        group_ids: List[int] = SchemaField(
            description="List of group IDs to send the campaign to",
            placeholder="e.g., ['1234567', '7654321']",
        )
        reply_to: str = SchemaField(
            description="Reply-to email address (must be verified in MailerLite)",
            placeholder="e.g., replies@company.com",
            default="",
        )

    class Output(BlockSchema):
        campaign_id: str = SchemaField(description="ID of the created campaign")
        created_campaign_name: str = SchemaField(description="Name of the created campaign")
        status: str = SchemaField(description="Status of the campaign (e.g., draft)")
        scheduled_for: str = SchemaField(description="When the campaign is scheduled to be sent")
        language_id: str = SchemaField(description="Language ID used for the campaign")
        track_opens: str = SchemaField(description="Whether email opens are being tracked")
        use_analytics: str = SchemaField(description="Whether Google Analytics is enabled")
        email_stats: str = SchemaField(description="Statistics about email performance as JSON string")
        created_at: str = SchemaField(description="When the campaign was created")
        error: str = SchemaField(description="Error message if the operation failed")

    def __init__(self):
        super().__init__(
            id="543d4ef3-b5b6-4428-bbcd-a2e15f06cc27",
            description="Create a new email campaign in MailerLite",
            categories={BlockCategory.COMMUNICATION},
            input_schema=MailerLiteCampaignBlock.Input,
            output_schema=MailerLiteCampaignBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "campaign_name": "Test Campaign",
                "subject": "Test Subject",
                "from_name": "Test Sender",
                "from_email": "test@example.com",
                "content": "Test content",
                "group_ids": ["1234567"],
                "reply_to": "replies@example.com",
            },
            test_output=[
                (
                    "campaign_id",
                    "1",
                ),
                (
                    "created_campaign_name",
                    "Test Campaign",
                ),
                (
                    "status",
                    "draft",
                ),
                (
                    "scheduled_for",
                    "2022-07-26 15:11:51",
                ),
                (
                    "language_id",
                    "4",
                ),
                (
                    "track_opens",
                    True,
                ),
                (
                    "use_analytics",
                    False,
                ),
                (
                    "email_stats",
                    {
                        "sent": 10,
                        "opens_count": 0,
                        "unique_opens_count": 0,
                        "open_rate": 0,
                        "clicks_count": 0,
                        "unique_clicks_count": 0,
                        "click_rate": 0,
                        "unsubscribes_count": 0,
                        "spam_count": 0,
                        "bounces_count": 0,
                    },
                ),
                (
                    "created_at",
                    "2022-07-26 15:07:52",
                ),
            ],
            test_mock={
                "_create_campaign": lambda *args, **kwargs: {
                    "data": {
                        "id": "1",
                        "name": "Test Campaign",
                        "status": "draft",
                        "scheduled_for": "2022-07-26 15:11:51",
                        "language_id": "4",
                        "settings": {
                            "track_opens": True,
                            "use_google_analytics": False,
                        },
                        "created_at": "2022-07-26 15:07:52",
                        "emails": [{
                            "stats": {
                                "sent": 10,
                                "opens_count": 0,
                                "unique_opens_count": 0,
                                "open_rate": {"float": 0},
                                "clicks_count": 0,
                                "unique_clicks_count": 0,
                                "click_rate": {"float": 0},
                                "unsubscribes_count": 0,
                                "spam_count": 0,
                                "hard_bounces_count": 0,
                                "soft_bounces_count": 0,
                            }
                        }]
                    }
                }
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def _create_campaign(self, client: mailerlite.Client, input_data: Input) -> dict:
        params = {
            "name": input_data.campaign_name,
            "language_id": 1,  # English
            "type": "regular",
            "groups": input_data.group_ids,
            "emails": [{
                "subject": input_data.subject,
                "from_name": input_data.from_name,
                "from": input_data.from_email,
                "content": input_data.content,
            }]
        }
        if input_data.reply_to:
            params["emails"][0]["reply_to"] = input_data.reply_to
        
        return client.campaigns.create(params)

    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs) -> BlockOutput:
        try:
            client = mailerlite.Client({
                'api_key': credentials.api_key.get_secret_value()
            })
            
            response = self._create_campaign(client, input_data)
            
            if isinstance(response, dict):
                if 'data' in response:
                    campaign_data = response['data']
                else:
                    campaign_data = response  # Some API endpoints return data directly
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
                
            
            # Convert email stats to JSON string
            email_stats = {
                "sent": 0,
                "opens_count": 0,
                "unique_opens_count": 0,
                "open_rate": 0.0,
                "clicks_count": 0,
                "unique_clicks_count": 0,
                "click_rate": 0.0,
                "unsubscribes_count": 0,
                "spam_count": 0,
                "bounces_count": 0
            }
            
            if campaign_data.get("emails") and isinstance(campaign_data["emails"], list):
                first_email = campaign_data["emails"][0]
                stats = first_email.get("stats", {})
                
                if isinstance(stats, dict):
                    email_stats.update({
                        "sent": int(stats.get("sent", 0)),
                        "opens_count": int(stats.get("opens_count", 0)),
                        "unique_opens_count": int(stats.get("unique_opens_count", 0)),
                        "open_rate": float(
                            stats.get("open_rate", {}).get("float", 0)
                            if isinstance(stats.get("open_rate"), dict)
                            else 0
                        ),
                        "clicks_count": int(stats.get("clicks_count", 0)),
                        "unique_clicks_count": int(stats.get("unique_clicks_count", 0)),
                        "click_rate": float(
                            stats.get("click_rate", {}).get("float", 0)
                            if isinstance(stats.get("click_rate"), dict)
                            else 0
                        ),
                        "unsubscribes_count": int(stats.get("unsubscribes_count", 0)),
                        "spam_count": int(stats.get("spam_count", 0)),
                        "bounces_count": int(
                            stats.get("hard_bounces_count", 0) + 
                            stats.get("soft_bounces_count", 0)
                        )
                    })
            
            settings = campaign_data.get("settings", {})
            
            yield "campaign_id", str(campaign_data.get("id", ""))
            yield "created_campaign_name", str(campaign_data.get("name", ""))
            yield "status", str(campaign_data.get("status", "draft"))
            yield "scheduled_for", str(campaign_data.get("scheduled_for", ""))
            yield "language_id", str(campaign_data.get("language_id", ""))
            yield "track_opens", str(settings.get("track_opens", True))
            yield "use_analytics", str(settings.get("use_google_analytics", False))
            yield "email_stats", json.dumps(email_stats)
            yield "created_at", str(campaign_data.get("created_at", ""))

        except Exception as e:
            yield "error", str(e)