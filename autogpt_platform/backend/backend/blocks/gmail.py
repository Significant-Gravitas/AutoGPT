import base64
import json
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

GOOGLE_CLIENT_ID = "39760950846-nvcrmjjdmm3f489k2tmrgt4lhpvsk09c.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_CLIENT_TOKEN_URI = "https://oauth2.googleapis.com/token"
GOOGLE_CLIENT_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

class GmailBlock(Block):
    class Input(BlockSchema):
        email: str = SchemaField(
            description="Email address that authorizes sending emails"
        )
        subject_text = SchemaField(
            description="Subject of the email",
            default="AutoGPT Notification"
        )
        body_text: str = SchemaField(
            description="Body of the email"
        )
        creds: str = SchemaField(
            description="JSON text of Gmail credentials"
        )

    class Output(BlockSchema):
        status: bool = SchemaField(description="Status of the email sending operation")
        error: str = SchemaField(
            description="Error message if the email sending failed"
        )

    def __init__(self):
        super().__init__(
            disabled=True,
            id="eaf81b28-b42f-47e7-9a14-5af3edfa4b1e",
            description="This block sends an email using the Gmail credentials.",
            categories={BlockCategory.OUTPUT},
            input_schema=GmailBlock.Input,
            output_schema=GmailBlock.Output,
            test_input={
                "email": "tuan.nguyen930708@gmail.com",
                "subject_text": "AutoGPT Notification",
                "body_text": "Sending by AutoGPT",
                "creds": "",
            },
            test_output=[("status", "OK")],
            test_mock={"send_email": lambda *args, **kwargs: "OK"},
        )

    @staticmethod
    def send_email(email: str, subject_text:str, body_text: str, creds: str) -> str:
        raw = json.loads(creds)
        credentials = Credentials(
            token=raw["access_token"],
            refresh_token=raw["refresh_token"],
            token_uri=GOOGLE_CLIENT_TOKEN_URI,
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=GOOGLE_CLIENT_SCOPES
        )   
        service = build("gmail", "v1", credentials=credentials)
        
        # Create your email content (plaintext in this example)
        mime_message = MIMEText(body_text)
        
        # Set the appropriate headers
        # "from" can be dynamically derived from the credentials if desired,
        # but in many cases you can simply specify the user’s known email address.
        mime_message["from"] = email
        mime_message["to"] = email
        mime_message["subject"] = subject_text

        # Convert the MIMEText object to a base64-encoded string (URL-safe)
        raw_string = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()

        # Prepare the payload for Gmail API
        payload = {
            "raw": raw_string
        }
        
        # Use the API to send the email from the authorized user's account
        return service.users().messages().send(
            userId="me",  # "me" indicates the authorized user
            body=payload
        ).execute()

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            response = self.send_email(
                input_data.email,
                input_data.subject_text,
                input_data.body_text,
                input_data.creds,
            )
            print(response)
            yield "status", "OK"   
        except Exception as e:
            yield "status", "KO"
            yield "error", f"An error occurred: {e}"   
