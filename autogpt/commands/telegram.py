
import requests
from autogpt.config import Config

cfg = Config()
telegram_token = cfg.telegram_token
telegram_chat_id = cfg.telegram_chat_id

def send_telegram_message(text):
    url = f'https://api.telegram.org/bot{telegram_token}/sendMessage'
    payload = {
                'chat_id': telegram_chat_id,
                'text': text
                }
   
    r = requests.post(url,json=payload)

    return r

def send_telegram_photo(photo):
    url = f'https://api.telegram.org/bot{telegram_token}/sendPhoto'
    payload = {
        'chat_id': telegram_chat_id,
        'photo': photo
    }
 
    r = requests.post(url, json=payload)
    return r

def send_telegram_document(document):
    url = f'https://api.telegram.org/bot{telegram_token}/sendDocument'
    payload = {
        'chat_id': telegram_chat_id,
        'document': document
    }
 
    r = requests.post(url, json=payload)
    return r

def send_telegram_video(video):
    url = f'https://api.telegram.org/bot{telegram_token}/sendVideo'
    payload = {
        'chat_id': telegram_chat_id,
        'video': video
    }
 
    r = requests.post(url, json=payload)
    return r

def send_telegram_audio(audio):
    url = f'https://api.telegram.org/bot{telegram_token}/sendVoice'
    payload = {
        'chat_id': telegram_chat_id,
        'voice': audio
    }
 
    r = requests.post(url, json=payload)
    return r

def send_telegram_location(location):
    url = f'https://api.telegram.org/bot{telegram_token}/sendLocation'
    payload = {
        'chat_id': telegram_chat_id,
        'voice': location
    }
 
    r = requests.post(url, json=payload)
    return r