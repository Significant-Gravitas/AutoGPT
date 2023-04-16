import asyncio
from queue import Queue

from autogpt.config.config import Config
from telegram import Bot, Update
from telegram.ext import CallbackContext, MessageHandler, filters
import asyncio
import traceback

cfg = Config()
response_queue = ""


def is_authorized_user(update: Update):
    return update.effective_user.id == int(cfg.telegram_chat_id)


async def delete_old_messages():
    bot = TelegramUtils.get_bot()
    updates = await bot.get_updates(offset=0)
    count = 0
    for update in updates:
        try:
            print("Deleting message: " + update.message.text + " " + str(update.message.message_id))
            await bot.delete_message(chat_id=update.message.chat.id, message_id=update.message.message_id)
        except Exception as e:
            print(f"Error while deleting message: {e} \n update: {update} \n {traceback.format_exc()}")
        count += 1
    if (count > 0):
        print("Cleaned up old messages.")


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
        print ("Sending message to Telegram.. ")
        recipient_chat_id = cfg.telegram_chat_id
        bot = TelegramUtils.get_bot()
        await bot.send_message(chat_id=recipient_chat_id, text=message)

    @staticmethod
    async def ask_user_async(prompt):
        global response_queue
        question = prompt + " (reply to this message)"

        #await delete_old_messages()

        print("Asking user: " + question)
        TelegramUtils.send_message(question)

        print("Waiting for response on Telegram chat...")
        await TelegramUtils._poll_updates()
        response_text = response_queue

        print("Response received from Telegram: " + response_text)
        return response_text

    @staticmethod
    async def _poll_updates():
        global response_queue
        bot = TelegramUtils.get_bot()

        last_update = await bot.get_updates(timeout=30)
        last_update_id = last_update[-1].update_id

        while True:
            try:
                print("Polling updates...")
                updates = await bot.get_updates(offset=last_update_id + 1, timeout=30)
                for update in updates:
                    if update.message and update.message.text:
                        if is_authorized_user(update):
                            response_queue =update.message.text
                            return
                    last_update_id = max(last_update_id, update.update_id)
            except Exception as e:
                print(f"Error while polling updates: {e}")

            await asyncio.sleep(1)
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
