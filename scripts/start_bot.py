import asyncio
import main

from telegram import Bot
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters
from telegram_chat import TelegramUtils, is_authorized_user, handle_response, Update
import os
from config import Config

cfg = Config()

main_started = False

application = Application.builder().token(cfg.telegram_api_key).build()


async def stop(update: Update, context: CallbackContext):
    if is_authorized_user(update):
        await update.message.reply_text("Stopping Auto-GPT now!")
        exit(0)


async def delete_old_messages():
    async with Bot(cfg.telegram_api_key) as bot:
        updates = await bot.get_updates()
        for update in updates:
            await bot.delete_message(chat_id=cfg.telegram_chat_id, message_id=update.message.message_id)
    print("Cleaned up old messages.")


async def start(update: Update, context: CallbackContext):
    global main_started
    print("Starting Auto-GPT...")
    if is_authorized_user(update):
        if (main_started):
            await update.message.reply_text("Already started!")

        else:
            main_started = True
            await TelegramUtils.send_message("Auto-GPT is starting now!")
            application.add_handler(CommandHandler("stop", stop))
            os.system("python3 ./main.py --gpt3only --speak ")


async def catch_all(update: Update, context: CallbackContext):
    print("Received message: " + update.message.text)
    if is_authorized_user(update):
        await TelegramUtils.send_message("I'm sorry, I was offline and missed your message. Please try again.")


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Delete old messages
    loop.run_until_complete(delete_old_messages())

    loop.run_until_complete(TelegramUtils().send_message(
        "Hello! I need you to start Auto-GPT for me. \n Please type /start."))

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, handle_response))
    application.add_handler(MessageHandler(filters.ALL, catch_all))

    try:
        print("Listening to Telegram...")
        loop.run_until_complete(application.run_polling())
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    main()
