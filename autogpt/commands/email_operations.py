import email
import imaplib
import os
import smtplib
from email.mime.text import MIMEText

from autogpt.commands.command import command


# Sending email
@command(
    "send_email",
    "Send an email",
    '"recipient": "<recipient_email>", "subject": "<subject>", "body": "<body>"',
)
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    email_sender = os.environ.get("EMAIL_SENDER")
    email_password = os.environ.get("EMAIL_PASSWORD")

    # Set up the email
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = email_sender
    msg["To"] = recipient

    # Send the email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_sender, email_password)
            server.sendmail(email_sender, recipient, msg.as_string())
        return f"Email sent successfully to {recipient}!"
    except Exception as e:
        return f"Error sending email: {str(e)}"


# Receiving email
@command(
    "receive_email",
    "Receive emails",
    '"folder": "<folder>"',
)
def receive_email(folder: str) -> str:
    """Receive emails from a folder."""
    email_sender = os.environ.get("EMAIL_SENDER")
    email_password = os.environ.get("EMAIL_PASSWORD")

    # Connect to the email server
    try:
        with imaplib.IMAP4_SSL("imap.gmail.com") as mail:
            mail.login(email_sender, email_password)
            mail.select(folder)

            # Search for all emails in the folder
            _, message_numbers = mail.search(None, "ALL")
            message_numbers = message_numbers[0].split()

            # Process and print emails
            for num in message_numbers[-10:]:  # Get the last 10 emails
                _, msg_data = mail.fetch(num, "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])

                print(f"From: {msg['From']}")
                print(f"Subject: {msg['Subject']}")
                print(f"Date: {msg['Date']}")
                print("")

            mail.close()
            mail.logout()

        return f"Processed the last 10 emails from the {folder} folder."
    except Exception as e:
        return f"Error receiving emails: {str(e)}"
