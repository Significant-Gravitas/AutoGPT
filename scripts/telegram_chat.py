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
response_received = False
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
        print("response_queue: " + str(response_queue))

        if is_authorized_user(update):
            response_queue.put(update.message.text)
            response_received = True
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
    async def ask_user(question):
        global response_queue

        await TelegramUtils._send_message(question)

        print("Waiting for response...")
        while (response_queue.empty()):
            await asyncio.sleep(0.5)
        response_text = await response_queue.get()
        print("Response received: " + response_text)
        return response_text


async def wait_for_response():
    global response_received, response_text
    while not response_received:
        await asyncio.sleep(0.1)
    return response_text
