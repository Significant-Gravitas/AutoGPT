import asyncio
import main

from telegram.ext import CommandHandler, MessageHandler, CallbackContext, Application
from telegram.ext import filters
from telegram_chat import TelegramUtils, application, is_authorized_user, handle_response, Update
import os

main_started = False

async def start(update: Update, context: CallbackContext):
    global main_started
    if is_authorized_user(update):
        if(main_started):
            await update.message.reply_text("Already started!")
        else:
            main_started = True
            await update.message.reply_text("Starting Auto-GPT now!")
            os.system("python3 ./main.py --gpt3only --speak ")


def main():
    print("Starting up...")
    asyncio.run(TelegramUtils().send_message("Hello! I need you to confirm with /start to start me. <3"))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, handle_response))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(application.run_polling())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
