from telegram import Update, Bot
from telegram.ext import CommandHandler, MessageHandler, CallbackContext, Application
from telegram.ext import filters
from config import Config
import threading
import asyncio
from functools import wraps

from threading import Lock, Semaphore

cfg = Config()
response_received = threading.Event()
response_text = ""

mutex_lock = Lock()  # Ensure only one sound is played at a time
# The amount of sounds to queue before blocking the main thread
queue_semaphore = Semaphore(1)

application = Application.builder().token(cfg.telegram_api_key).build()


def is_authorized_user(update: Update):
    return update.effective_user.id == int(cfg.telegram_chat_id)


def handle_response(update: Update, context: CallbackContext):
    global response_received, response_text

    if is_authorized_user(update):
        response_text = update.message.text
        response_received.set()


class TelegramUtils:
    @staticmethod
    def send_message(message):
        async def _send_message_async():
            bot_token = cfg.telegram_api_key
            recipient_chat_id = cfg.telegram_chat_id
            await Bot(bot_token).send_message(chat_id=recipient_chat_id, text=message)
        asyncio.run(_send_message_async())

    @staticmethod
    def ask_user(question):
        global response_received, response_text

        response_received.clear()
        response_text = ""

        TelegramUtils.send_message(question)

        print("Waiting for response...")
        response_received.wait()
        return response_text


class SignalSafeEventLoop(asyncio.SelectorEventLoop):
    def add_signal_handler(self, sig, callback, *args):
        pass

    def remove_signal_handler(self, sig):
        pass


def run_in_signal_safe_loop(coro):
    @wraps(coro)
    def wrapper(*args, **kwargs):
        loop = SignalSafeEventLoop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


def main():
    application.add_handler(MessageHandler(filters.TEXT, handle_response))
    application.run_polling()
