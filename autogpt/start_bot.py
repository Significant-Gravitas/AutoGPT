import asyncio
import os
import sys
import traceback
from threading import Lock, Semaphore

from telegram import Update
from telegram.ext import Application, CallbackContext, CommandHandler

try:
    from autogpt.config.config import Config
    from autogpt.telegram_chat import TelegramUtils, is_authorized_user
except ModuleNotFoundError:
    from config import Config
    from telegram_chat import TelegramUtils, is_authorized_user

cfg = Config()

main_started = False

mutex_lock = Lock()  # Ensure only one sound is played at a time
# The amount of sounds to queue before blocking the main thread
queue_semaphore = Semaphore(1)


async def stop(update: Update, context: CallbackContext):
    if is_authorized_user(update):
        process = os.popen('ps -Af')
        output = process.read()
        process.close()
        if "python -m autogpt" in output:
            await update.message.reply_text("Stopping Auto-GPT now!")
            os.system("pkill -f autogpt")
            TelegramUtils.send_message("Auto-GPT was stopped.")
        else:
            TelegramUtils.send_message("Auto-GPT was not running.")


async def start(update: Update, context: CallbackContext):
    global main_started
    print("Starting Auto-GPT...")
    if is_authorized_user(update):
        # Check if operating system is Windows
        if os.name == 'nt':
            # Check if main is still running
            process = os.popen('tasklist')
        else:
            # check if main is still running
            process = os.popen('ps -Af')
        output = process.read()
        process.close()
        if "python -m autogpt" in output:
            await update.message.reply_text("Already started!")
            return
        else:
            TelegramUtils.send_message("Auto-GPT is starting now!")
            os.system("python -m autogpt {}".format(" ".join(sys.argv[1:])))


def main():
    print("Starting up...")

    telegramUtils = TelegramUtils()
    telegramUtils.send_message("Starting Auto-GPT...")

    # Delete old messages
    asyncio.run(telegramUtils.delete_old_messages())

    TelegramUtils().send_message(
        "Hello! I need you to confirm with /start to start me. <3")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application = Application.builder().token(cfg.telegram_api_key).build()
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("start", start))
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
