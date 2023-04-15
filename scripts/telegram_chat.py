import asyncio
from queue import Queue

from config import Config
from telegram import Bot, Update
from telegram.ext import CallbackContext, MessageHandler, filters
import asyncio

cfg = Config()
response_queue = asyncio.Queue()


def is_authorized_user(update: Update):
    return update.effective_user.id == int(cfg.telegram_chat_id)


def handle_response(update: Update, context: CallbackContext):
    try:
        print("Received response: " + update.message.text)

        if is_authorized_user(update):
            response_queue.put(update.message.text)
    except Exception as e:
        print(e)


class TelegramUtils:
    @staticmethod
    def get_bot():
        bot_token = cfg.telegram_api_key
        return Bot(bot_token)

    @staticmethod
    def send_message(message):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 'RuntimeError: There is no current event loop...'
            loop = None
        if loop and loop.is_running():
            loop.create_task(TelegramUtils._send_message(message))
        else:
            asyncio.run(TelegramUtils._send_message(message))

    @staticmethod
    async def _send_message(message):
        recipient_chat_id = cfg.telegram_chat_id
        bot = TelegramUtils.get_bot()
        await bot.send_message(chat_id=recipient_chat_id, text=message)

    @staticmethod
    async def ask_user_async(prompt):
        question = prompt + " (reply to this message)"
        print("Asking user: " + question)
        TelegramUtils.send_message(question)

        print("Waiting for response...")
        response_text = await response_queue.get()

        print("Response received: " + response_text)
        return response_text

    @staticmethod
    def ask_user(prompt):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 'RuntimeError: There is no current event loop...'
            loop = None
        if loop and loop.is_running():
            return loop.create_task(TelegramUtils.ask_user_async(prompt))
        else:
            return asyncio.run(TelegramUtils.ask_user_async(prompt))
