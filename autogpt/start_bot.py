import asyncio
import os
import threading
import sys
import traceback
from threading import Lock, Semaphore

from telegram import Update
from telegram.ext import Application, CallbackContext, CommandHandler, filters, MessageHandler
from autogpt.config.config import Config


from autogpt.telegram_chat import TelegramUtils, handle_response, is_authorized_user

cfg = Config()

main_started = False

application = Application.builder().token(cfg.telegram_api_key).build()

mutex_lock = Lock()  # Ensure only one sound is played at a time
# The amount of sounds to queue before blocking the main thread
queue_semaphore = Semaphore(1)


async def stop(update: Update, context: CallbackContext):
    if is_authorized_user(update):
        await update.message.reply_text("Stopping Auto-GPT now!")
        exit(0)


async def delete_old_messages():
    bot = TelegramUtils.get_bot()
    updates = await bot.get_updates(offset=0)
    count = 0
    for update in updates:
        print("Deleting message: " + update.message.text)
        await bot.delete_message(chat_id=cfg.telegram_chat_id, message_id=update.message.message_id)
        count += 1
    if (count > 0):
        print("Cleaned up old messages.")


async def start(update: Update, context: CallbackContext):
    global main_started
    print("Starting Auto-GPT...")
    if is_authorized_user(update):
        if main_started:
            TelegramUtils.send_message("Already started!")
        else:
            main_started = True
            TelegramUtils.send_message("Auto-GPT is starting now!")
            os.system("python -m autogpt {}".format(" ".join(sys.argv[1:])))


def main():
    print("Starting up...")

    # Delete old messages
    asyncio.run(delete_old_messages())

    TelegramUtils().send_message(
        "Hello! I need you to confirm with /start to start me. <3")

    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, handle_response))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        print("Check your Telegram chat to start Auto-GPT! ;-)")
        loop.run_until_complete(application.run_polling())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
        # queue_semaphore.acquire(True)
        # thread = threading.Thread(target=main)
        # thread.start()
    except KeyboardInterrupt:
        print("Exiting...")
        TelegramUtils.send_message(
            "I hope I could help! :) \n \n Bye bye! <3")

        exit(0)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        TelegramUtils.send_message(
            "Sorry, I have to stop. \n \n An error occurred: " + str(e))
        exit(1)
