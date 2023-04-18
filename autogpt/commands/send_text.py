import os
from dotenv import load_dotenv
from twilio.rest import Client

# Load variables from .env file
load_dotenv()

# Get Twilio Account SID, Auth Token, and phone number from environment variables
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
from_number = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize the Twilio client
client = Client(account_sid, auth_token)

def send_sms(to, body):
    message = client.messages.create(
        body=body,
        from_=from_number,
        to=to
    )

    print(f'Sent message: {message.sid}')

if __name__ == '__main__':
    # Replace this with the recipient's phone number (including country code)
    recipient = '+11234567890'
    message_body = 'Hello, this is a message from my Python application!'
    
    send_sms(recipient, message_body)
