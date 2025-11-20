import smtplib
import socket
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)
from backend.integrations.providers import ProviderName

TEST_CREDENTIALS = UserPasswordCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="smtp",
    username=SecretStr("mock-smtp-username"),
    password=SecretStr("mock-smtp-password"),
    title="Mock SMTP credentials",
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
SMTPCredentials = UserPasswordCredentials
SMTPCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.SMTP],
    Literal["user_password"],
]


def SMTPCredentialsField() -> SMTPCredentialsInput:
    return CredentialsField(
        description="The SMTP integration requires a username and password.",
    )


class SMTPConfig(BaseModel):
    smtp_server: str = SchemaField(description="SMTP server address")
    smtp_port: int = SchemaField(default=25, description="SMTP port number")

    model_config = ConfigDict(title="SMTP Config")


class SendEmailBlock(Block):
    class Input(BlockSchemaInput):
        to_email: str = SchemaField(
            description="Recipient email address", placeholder="recipient@example.com"
        )
        subject: str = SchemaField(
            description="Subject of the email", placeholder="Enter the email subject"
        )
        body: str = SchemaField(
            description="Body of the email", placeholder="Enter the email body"
        )
        config: SMTPConfig = SchemaField(description="SMTP Config")
        credentials: SMTPCredentialsInput = SMTPCredentialsField()

    class Output(BlockSchemaOutput):
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
                "config": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 25,
                },
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("status", "Email sent successfully")],
            test_mock={"send_email": lambda *args, **kwargs: "Email sent successfully"},
        )

    @staticmethod
    def send_email(
        config: SMTPConfig,
        to_email: str,
        subject: str,
        body: str,
        credentials: SMTPCredentials,
    ) -> str:
        smtp_server = config.smtp_server
        smtp_port = config.smtp_port
        smtp_username = credentials.username.get_secret_value()
        smtp_password = credentials.password.get_secret_value()

        msg = MIMEMultipart()
        msg["From"] = smtp_username
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            # Try to connect to the SMTP server
            try:
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            except socket.gaierror:
                raise ConnectionError(
                    f"Cannot connect to SMTP server '{smtp_server}'. "
                    "Please verify the server address is correct."
                )
            except socket.timeout:
                raise ConnectionError(
                    f"Connection timeout to '{smtp_server}' on port {smtp_port}. "
                    "The server may be down or unreachable."
                )
            except (ConnectionRefusedError, OSError) as e:
                raise ConnectionError(
                    f"Connection refused to '{smtp_server}' on port {smtp_port}. "
                    f"Common SMTP ports are: 587 (TLS), 465 (SSL), 25 (plain). "
                    f"Error: {str(e)}"
                )

            try:
                # Start TLS encryption
                try:
                    server.starttls()
                except smtplib.SMTPNotSupportedError:
                    server.quit()
                    raise ConnectionError(
                        f"STARTTLS not supported by server '{smtp_server}'. "
                        "Try using port 465 for SSL or port 25 for unencrypted connection."
                    )
                except ssl.SSLError as e:
                    server.quit()
                    raise ConnectionError(
                        f"SSL/TLS error when connecting to '{smtp_server}': {str(e)}. "
                        "The server may require a different security protocol."
                    )

                # Authenticate
                try:
                    server.login(smtp_username, smtp_password)
                except smtplib.SMTPAuthenticationError:
                    server.quit()
                    raise ConnectionError(
                        "Authentication failed. Please verify your username and password are correct."
                    )
                except smtplib.SMTPException as e:
                    server.quit()
                    raise ConnectionError(
                        f"Authentication error: {str(e)}. "
                        "Please check your credentials and server settings."
                    )

                # Send email
                try:
                    server.sendmail(smtp_username, to_email, msg.as_string())
                except smtplib.SMTPRecipientsRefused:
                    server.quit()
                    raise ValueError(
                        f"Recipient email address '{to_email}' was rejected by the server. "
                        "Please verify the email address is valid."
                    )
                except smtplib.SMTPSenderRefused:
                    server.quit()
                    raise ValueError(
                        f"Sender email address '{smtp_username}' was rejected by the server. "
                        "Please verify your account is authorized to send emails."
                    )
                except smtplib.SMTPDataError as e:
                    server.quit()
                    raise ValueError(f"Email data rejected by server: {str(e)}")

                server.quit()
            except Exception:
                # Ensure server connection is closed on any error
                try:
                    server.quit()
                except Exception:
                    pass
                raise

        except (ConnectionError, ValueError):
            # Re-raise our custom error messages
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise RuntimeError(
                f"Unexpected error sending email: {type(e).__name__}: {str(e)}"
            )

        return "Email sent successfully"

    async def run(
        self, input_data: Input, *, credentials: SMTPCredentials, **kwargs
    ) -> BlockOutput:
        try:
            status = self.send_email(
                config=input_data.config,
                to_email=input_data.to_email,
                subject=input_data.subject,
                body=input_data.body,
                credentials=credentials,
            )
            yield "status", status
        except Exception as e:
            # We need to catch the error and yield it as error 
            # To trigger the BlockExecutionError wrapper
            yield "error", str(e)
