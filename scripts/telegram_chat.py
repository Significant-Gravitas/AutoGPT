import asyncio
import threading
from functools import wraps
from threading import Lock, Semaphore
import time
from queue import Queue

from config import Config
from telegram import Bot, Update
from telegram.ext import (Application, CommandHandler,
                          CallbackContext, MessageHandler, filters)

cfg = Config()
response_received = asyncio.Event()
response_text = ""
response_queue = Queue()

mutex_lock = Lock()  # Ensure only one sound is played at a time
# The amount of sounds to queue before blocking the main thread
queue_semaphore = Semaphore(1)


def is_authorized_user(update: Update):
    return update.effective_user.id == int(cfg.telegram_chat_id)


def handle_response(update: Update, context: CallbackContext):
    try:
        print("Received response: " + update.message.text)

        if is_authorized_user(update):
            global response_text
            response_text = update.message.text
            response_received.set()
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
    async def _ask_user(question):
        global response_queue

        await TelegramUtils._send_message(question)

        print("Waiting for response...")
        response_text = await (response_received.wait())

        print("Response received: " + response_text)
        return response_text

    @staticmethod
    def ask_user(question):
        print("Asking user: " + question)
        try:
            loop = asyncio.get_running_loop()
            print("Asking user: " + question)
        except RuntimeError:
            loop = None
            print("No running loop")
        if loop and loop.is_running():
            print("loop is running")
            return loop.create_task(TelegramUtils._ask_user(question))
        else:
            print("loop is not running")
            return asyncio.run(TelegramUtils._ask_user(question))


async def wait_for_response():
    global response_received, response_text
    while not response_received:
        await asyncio.sleep(0.1)
    return response_text
