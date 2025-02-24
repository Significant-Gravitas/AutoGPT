from typing import Literal

from pydantic import BaseModel


# Models from https://account.postmarkapp.com/servers/<id>/streams/outbound/webhooks/new
class PostmarkDeliveryWebhook(BaseModel):
    RecordType: Literal["Delivery"] = "Delivery"
    ServerID: int
    MessageStream: str
    MessageID: str
    Recipient: str
    Tag: str
    DeliveredAt: str
    Details: str
    Metadata: dict[str, str]


class PostmarkBounceWebhook(BaseModel):
    RecordType: Literal["Bounce"] = "Bounce"
    ID: int
    Type: str
    TypeCode: int
    Tag: str
    MessageID: str
    Details: str
    Email: str
    From: str
    BouncedAt: str
    Inactive: bool
    DumpAvailable: bool
    CanActivate: bool
    Subject: str
    ServerID: int
    MessageStream: str
    Content: str
    Name: str
    Description: str
    Metadata: dict[str, str]


class PostmarkSpamComplaintWebhook(BaseModel):
    RecordType: Literal["SpamComplaint"] = "SpamComplaint"
    ID: int
    Type: str
    TypeCode: int
    Tag: str
    MessageID: str
    Details: str
    Email: str
    From: str
    BouncedAt: str
    Inactive: bool
    DumpAvailable: bool
    CanActivate: bool
    Subject: str
    ServerID: int
    MessageStream: str
    Content: str
    Name: str
    Description: str
    Metadata: dict[str, str]


class PostmarkOpenWebhook(BaseModel):
    RecordType: Literal["Open"] = "Open"
    MessageStream: str
    Metadata: dict[str, str]
    FirstOpen: bool
    Recipient: str
    MessageID: str
    ReceivedAt: str
    Platform: str
    ReadSeconds: int
    Tag: str
    UserAgent: str
    OS: dict[str, str]
    Client: dict[str, str]
    Geo: dict[str, str]


class PostmarkClickWebhook(BaseModel):
    RecordType: Literal["Click"] = "Click"
    MessageStream: str
    Metadata: dict[str, str]
    Recipient: str
    MessageID: str
    ReceivedAt: str
    Platform: str
    ClickLocation: str
    OriginalLink: str
    Tag: str
    UserAgent: str
    OS: dict[str, str]
    Client: dict[str, str]
    Geo: dict[str, str]


class PostmarkSubscriptionChangeWebhook(BaseModel):
    RecordType: Literal["SubscriptionChange"] = "SubscriptionChange"
    MessageID: str
    ServerID: int
    MessageStream: str
    ChangedAt: str
    Recipient: str
    Origin: str
    SuppressSending: bool
    SuppressionReason: str
    Tag: str
    Metadata: dict[str, str]


PostmarkWebhook = (
    PostmarkDeliveryWebhook
    | PostmarkBounceWebhook
    | PostmarkSpamComplaintWebhook
    | PostmarkOpenWebhook
    | PostmarkClickWebhook
    | PostmarkSubscriptionChangeWebhook
)
