
from autogpt.config import Config
from autogpt.workspace import path_in_workspace
from builtins import open
import requests
import os

cfg = Config()
telegram_token = cfg.telegram_token
telegram_chat_id = cfg.telegram_chat_id


def send_telegram_message(text):
    url = f'https://api.telegram.org/bot{telegram_token}/sendMessage'
    payload = {
                'chat_id': telegram_chat_id,
                'text': text
                }

    response = requests.post(url, json=payload)

    return check_error(response)


def send_telegram_photo(photo):
    url = f'https://api.telegram.org/bot{telegram_token}/sendPhoto'

    photoFile = get_file(photo)

    response = requests.post(url, data={'chat_id': telegram_chat_id}, files={'photo': photoFile})

    return check_error(response)


def send_telegram_document(document):
    url = f'https://api.telegram.org/bot{telegram_token}/sendDocument'

    documentFile = get_file(document)

    response = requests.post(url, data={'chat_id': telegram_chat_id}, files={'document': documentFile})

    return check_error(response)


def send_telegram_video(video):
    url = f'https://api.telegram.org/bot{telegram_token}/sendVideo'

    videoFile = get_file(video)

    response = requests.post(url, data={'chat_id': telegram_chat_id}, files={'video': videoFile})

    return check_error(response)


def send_telegram_audio(audio):
    url = f'https://api.telegram.org/bot{telegram_token}/sendVoice'

    audioFile = get_file(audio)

    response = requests.post(url, data={'chat_id': telegram_chat_id}, files={'voice': audioFile})

    return check_error(response)


def send_telegram_location(latitude, longitude):
    url = f'https://api.telegram.org/bot{telegram_token}/sendLocation'
    payload = {
        'chat_id': telegram_chat_id,
        'latitude': latitude,
        'longitude': longitude
    }

    r = requests.post(url, json=payload)
    return r


def get_file(path):
    try:
        return open(path, "rb")
    except OSError:
        return open(path_in_workspace(path), "rb")


def check_error(response):
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        return f'Http Error: {errh.response.text}'
    except requests.exceptions.ConnectionError as errc:
        return f'Error Connecting: {errc.response.text}'
    except requests.exceptions.Timeout as errt:
        return f'Timeout Error: {errt.response.text}'
    except requests.exceptions.RequestException as err:
        return f'Error: {err.response.text}'
    return f'Success: {response.text}'
