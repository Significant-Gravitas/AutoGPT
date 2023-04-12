from telegram import Bot
from config import Config

cfg = Config()


def send_message(message):
    bot_token = cfg.telegram_api_key
    recipient_chat_id = cfg.telegram_chat_id
    bot = Bot(bot_token)
    bot.send_message(chat_id=recipient_chat_id, text=message)
