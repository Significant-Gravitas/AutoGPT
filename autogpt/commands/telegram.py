import os
from dotenv import load_dotenv
import requests

load_dotenv()

def send_telegram(message_text):

    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    get_ID = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    try:
        chat_id = requests.get(get_ID).json()['result']
        if len(chat_id) < 1:
            return "Sending telegram message failed, you need to start conversation with your bot first."

        else:
            chat_id = requests.get(get_ID).json()['result'][0]['message']['from']['id']
            message = message_text
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
            requests.get(url) # this sends the message
            return "Telegram message sent successfully."
    
    except:
        return "ERROR! Invalid telegram bot token. Make sure you entered correct token in your .env file."
        
   
