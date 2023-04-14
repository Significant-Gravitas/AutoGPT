import asyncio
import threading
from functools import wraps
from threading import Lock, Semaphore

from config import Config
from telegram import Bot, Update
from telegram.ext import (Application, CallbackContext, MessageHandler, filters)

cfg = Config()
response_received = threading.Event()
response_text = ""

mutex_lock = Lock()  # Ensure only one sound is played at a time
# The amount of sounds to queue before blocking the main thread
queue_semaphore = Semaphore(1)

application = Application.builder().token(cfg.telegram_api_key).build()


def is_authorized_user(update: Update):
    return update.effective_user.id == int(cfg.telegram_chat_id)


async def handle_response(update: Update, context: CallbackContext):
    global response_received, response_text
    try:
        mutex_lock.acquire()
        print("Received response: " + update.message.text)
        print("response_received: " + str(response_received))

        if is_authorized_user(update):
            response_text = update.message.text
            response_received.set()
    except Exception as e:
        print(e)


def start_listening():
    print("Listening to Telegram...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        application.add_handler(MessageHandler(filters.TEXT, handle_response))
        loop.run_until_complete(application.run_polling())
    except KeyboardInterrupt:
        pass


class TelegramUtils:
    @staticmethod
    async def send_message(message):
        bot_token = cfg.telegram_api_key
        recipient_chat_id = cfg.telegram_chat_id
        await Bot(bot_token).send_message(chat_id=recipient_chat_id, text=message)

    @staticmethod
    def ask_user(question):
        global response_received, response_text

        response_received.clear()
        response_text = ""

        asyncio.run(TelegramUtils().send_message(question))
        start_listening()

        print("Waiting for response...")
        response_received.wait()
        print("Response received: " + response_text)
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


async def main():
    prompt = TelegramUtils.ask_user("Hello! I need you to confirm with /start to start me. <3")
    print(prompt)


if __name__ == "__main__":
    asyncio.run(main())
