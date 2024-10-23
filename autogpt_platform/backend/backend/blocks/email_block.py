import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from pydantic import BaseModel, ConfigDict, Field

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


class EmailCredentials(BaseModel):
    smtp_server: str = Field(
        default="smtp.gmail.com", description="SMTP server address"
    )
    smtp_port: int = Field(default=25, description="SMTP port number")
    smtp_username: BlockSecret = SecretField(key="smtp_username")
    smtp_password: BlockSecret = SecretField(key="smtp_password")

    model_config = ConfigDict(title="Email Credentials")


class SendEmailBlock(Block):
    class Input(BlockSchema):
        to_email: str = SchemaField(
            description="Recipient email address", placeholder="recipient@example.com"
        )
        subject: str = SchemaField(
            description="Subject of the email", placeholder="Enter the email subject"
        )
        body: str = SchemaField(
            description="Body of the email", placeholder="Enter the email body"
        )
        creds: EmailCredentials = Field(
            description="SMTP credentials",
            default=EmailCredentials(),
        )

    class Output(BlockSchema):
        status: str = SchemaField(description="Status of the email sending operation")
        error: str = SchemaField(
            description="Error message if the email sending failed"
        )

    def __init__(self):
        super().__init__(
            id="4335878a-394e-4e67-adf2-919877ff49ae",
            description="This block sends an email using the provided SMTP credentials.",
            categories={BlockCategory.OUTPUT},
            input_schema=SendEmailBlock.Input,
            output_schema=SendEmailBlock.Output,
            test_input={
                "to_email": "recipient@example.com",
                "subject": "Test Email",
                "body": "This is a test email.",
                "creds": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 25,
                    "smtp_username": "your-email@gmail.com",
                    "smtp_password": "your-gmail-password",
                },
            },
            test_output=[("status", "Email sent successfully")],
            test_mock={"send_email": lambda *args, **kwargs: "Email sent successfully"},
        )

    @staticmethod
    def send_email(
        creds: EmailCredentials, to_email: str, subject: str, body: str
    ) -> str:
        smtp_server = creds.smtp_server
        smtp_port = creds.smtp_port
        smtp_username = creds.smtp_username.get_secret_value()
        smtp_password = creds.smtp_password.get_secret_value()

        msg = MIMEMultipart()
        msg["From"] = smtp_username
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(smtp_username, to_email, msg.as_string())

        return "Email sent successfully"

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "status", self.send_email(
            input_data.creds,
            input_data.to_email,
            input_data.subject,
            input_data.body,
        )
