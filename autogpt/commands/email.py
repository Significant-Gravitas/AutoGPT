from autogpt.config import Config
import smtplib
from email.message import EmailMessage

CFG = Config()


def send_email(recipient: str, subject: str, message: str) -> str:
    """Send an email

    Args:
        recipient (str): The email of the recipients
        subject (str): The subject of the email
        message (str): The message content of the email

    Returns:
        str: Any error messages
    """
    host = CFG.email_smtp_host
    port = CFG.email_smtp_port
    sender = CFG.email_address
    sender_pwd = CFG.email_password

    if not sender or not sender_pwd:
        return f"Error: email not sent. EMAIL_ADDRESS or EMAIL_PASSWORD not set in environment."

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    msg.set_content(message)

    # send email
    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        smtp.login(sender, sender_pwd)
        smtp.send_message(msg)
