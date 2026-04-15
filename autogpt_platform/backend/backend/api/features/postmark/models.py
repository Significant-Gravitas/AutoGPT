from enum import Enum
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


class PostmarkBounceEnum(Enum):
    HardBounce = 1
    """
    The server was unable to deliver your message (ex: unknown user, mailbox not found).
    """
    Transient = 2
    """
    The server could not temporarily deliver your message (ex: Message is delayed due to network troubles).
    """
    Unsubscribe = 16
    """
    Unsubscribe or Remove request.
    """
    Subscribe = 32
    """
    Subscribe request from someone wanting to get added to the mailing list.
    """
    AutoResponder = 64
    """
    "Autoresponder" is an automatic email responder including nondescript NDRs and some "out of office" replies.
    """
    AddressChange = 128
    """
    The recipient has requested an address change.
    """
    DnsError = 256
    """
    A temporary DNS error.
    """
    SpamNotification = 512
    """
    The message was delivered, but was either blocked by the user, or classified as spam, bulk mail, or had rejected content.
    """
    OpenRelayTest = 1024
    """
    The NDR is actually a test email message to see if the mail server is an open relay.
    """
    Unknown = 2048
    """
    Unable to classify the NDR.
    """
    SoftBounce = 4096
    """
    Unable to temporarily deliver message (i.e. mailbox full, account disabled, exceeds quota, out of disk space).
    """
    VirusNotification = 8192
    """
    The bounce is actually a virus notification warning about a virus/code infected message.
    """
    ChallengeVerification = 16384
    """
    The bounce is a challenge asking for verification you actually sent the email. Typcial challenges are made by Spam Arrest, or MailFrontier Matador.
    """
    BadEmailAddress = 100000
    """
    The address is not a valid email address.
    """
    SpamComplaint = 100001
    """
    The subscriber explicitly marked this message as spam.
    """
    ManuallyDeactivated = 100002
    """
    The email was manually deactivated.
    """
    Unconfirmed = 100003
    """
    Registration not confirmed — The subscriber has not clicked on the confirmation link upon registration or import.
    """
    Blocked = 100006
    """
    Blocked from this ISP due to content or blacklisting.
    """
    SMTPApiError = 100007
    """
    An error occurred while accepting an email through the SMTP API.
    """
    InboundError = 100008
    """
    Processing failed — Unable to deliver inbound message to destination inbound hook.
    """
    DMARCPolicy = 100009
    """
    Email rejected due DMARC Policy.
    """
    TemplateRenderingFailed = 100010
    """
    Template rendering failed — An error occurred while attempting to render your template.
    """


class PostmarkBounceWebhook(BaseModel):
    RecordType: Literal["Bounce"] = "Bounce"
    ID: int
    Type: str
    TypeCode: PostmarkBounceEnum
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
