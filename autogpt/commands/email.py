import smtplib
import ssl
from email.message import EmailMessage

from autogpt.config import Config

CFG = Config()


email_password = CFG.email_password

email_sender = CFG.email_id


def send_email(email: str, subject : str, body : str ):

    email_receiver = email
    
    body = body
    subject=subject

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)
    
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())
        return "email sent successfully"