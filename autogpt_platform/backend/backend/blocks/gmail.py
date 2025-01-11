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
GOOGLE_CLIENT_SCOPES = ["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/gmail.send"]

class GmailBlock(Block):
    class Input(BlockSchema):
        email: str = SchemaField(
            description="Email address that authorizes sending emails",
            default=""
        )
        subject_text: str = SchemaField(
            description="Subject of the email",
            default="AutoGPT Notification"
        )
        body_text: str = SchemaField(
            description="Body of the email"
        )
        access_token: str = SchemaField(
            description="Gmail access token credentials"
        )

    class Output(BlockSchema):
        status: bool = SchemaField(description="Status of the email sending operation")
        error: str = SchemaField(
            description="Error message if the email sending failed"
        )

    def __init__(self):
        super().__init__(
            id="eaf81b28-b42f-47e7-9a14-5af3edfa4b1e",
            description="This block sends an email using the Gmail credentials.",
            categories={BlockCategory.OUTPUT},
            input_schema=GmailBlock.Input,
            output_schema=GmailBlock.Output,
            test_input={
                "email": "tuan.nguyen930708@gmail.com",
                "subject_text": "AutoGPT Notification",
                "body_text": "Sending by AutoGPT",
                "access_token": "",
            },
            test_output=[("status", "OK")],
            test_mock={"send_email": lambda *args, **kwargs: "OK"},
        )

    @staticmethod
    def send_email(email: str, subject_text:str, body_text: str, access_token: str) -> str:
        credentials = Credentials(
            token=access_token,
            # refresh_token=raw["refresh_token"],
            token_uri=GOOGLE_CLIENT_TOKEN_URI,
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=GOOGLE_CLIENT_SCOPES
        )   
        service = build("gmail", "v1", credentials=credentials)
        
        # Create your email content (plaintext in this example)
        mime_message = MIMEText(body_text)
        
        userinfo = GmailBlock.get_userinfo(access_token)
        # Set the appropriate headers
        # "from" can be dynamically derived from the credentials if desired,
        # but in many cases you can simply specify the user’s known email address.
        mime_message["from"] = userinfo["email"]
        mime_message["to"] = email if email else userinfo["email"] 
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
    
    def get_userinfo(access_token: str) -> dict :
        credentials = Credentials(
            token=access_token,
            # refresh_token=raw["refresh_token"],
            token_uri=GOOGLE_CLIENT_TOKEN_URI,
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=GOOGLE_CLIENT_SCOPES
        )
        service = build("oauth2", "v2", credentials=credentials)
        return service.userinfo().get().execute()
    
    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            response = self.send_email(
                input_data.email,
                input_data.subject_text,
                input_data.body_text,
                input_data.access_token,
            )
            #  response = {'id': '193fe2be6dae125c', 'threadId': '193fe2be6dae125c', 'labelIds': ['UNREAD', 'SENT', 'INBOX']}
            yield "status", True
        except Exception as e:
            yield "status", False
            yield "error", f"An error occurred: {e}"   
