import asyncio
try:
    from autogpt.config.config import Config
except ModuleNotFoundError:
    from config import Config
from telegram import Bot, Update
from telegram.ext import CallbackContext
import traceback

cfg = Config()
response_queue = ""


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
    async def delete_old_messages():
        bot = await TelegramUtils().get_bot()
        updates = await bot.get_updates(offset=0)
        count = 0
        for update in updates:
            try:
                print("Deleting message: " + update.message.text +
                      " " + str(update.message.message_id))
                await bot.delete_message(
                    chat_id=update.message.chat.id,
                    message_id=update.message.message_id
                )
            except Exception as e:
                print(
                    f"Error while deleting message: {e} \n"
                    + f" update: {update} \n {traceback.format_exc()}")
            count += 1
        if (count > 0):
            print("Cleaned up old messages.")

    @staticmethod
    async def get_bot():
        bot_token = cfg.telegram_api_key
        bot = Bot(token=bot_token)
        commands = await bot.get_my_commands()
        if len(commands) == 0:
            await TelegramUtils.set_commands(bot)
        commands = await bot.get_my_commands()
        return bot

    @staticmethod
    async def set_commands(bot):
        await bot.set_my_commands([
            ('start', 'Start Auto-GPT'),
            ('stop', 'Stop Auto-GPT'),
            ('help', 'Show help'),
            ('yes', 'Confirm'),
            ('no', 'Deny')
        ])

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
    def send_voice(voice_file):
        try:
            TelegramUtils.get_bot().send_voice(
                chat_id=cfg.telegram_chat_id, voice=open(voice_file, 'rb'))
        except RuntimeError:
            print("Error while sending voice message")

    @staticmethod
    async def _send_message(message):
        print("Sending message to Telegram.. ")
        recipient_chat_id = cfg.telegram_chat_id
        bot = await TelegramUtils.get_bot()
        await bot.send_message(chat_id=recipient_chat_id, text=message)

    @staticmethod
    async def ask_user_async(prompt):
        global response_queue
        question = prompt + " (reply to this message)"

        response_queue = ""
        # await delete_old_messages()

        print("Asking user: " + question)
        TelegramUtils.send_message(question)

        print("Waiting for response on Telegram chat...")
        await TelegramUtils._poll_updates()

        if response_queue == "/start":
            response_queue = await TelegramUtils.ask_user(
                "I am already here... \n Please use /stop to stop me first.")
        if response_queue == "/help":
            response_queue = await TelegramUtils.ask_user(
                "You can use /stop to stop me \n and /start to start me again.")

        if response_queue == "/stop":
            TelegramUtils.send_message("Stopping Auto-GPT now!")
            exit(0)
        elif response_queue == "/yes":
            response_text = "yes"
        elif response_queue == "/no":
            response_text = "no"
        if response_queue.capitalize() in [
            "Yes", "Okay", "Ok", "Sure", "Yeah", "Yup", "Yep"
        ]:
            response_text = "y"
        if response_queue.capitalize() in [
            "No", "Nope", "Nah", "N"
        ]:
            response_text = "n"
        else:
            response_text = response_queue

        print("Response received from Telegram: " + response_text)
        return response_text

    @staticmethod
    async def _poll_updates():
        global response_queue
        bot = await TelegramUtils.get_bot()

        last_update = await bot.get_updates(timeout=30)
        last_update_id = last_update[-1].update_id

        while True:
            try:
                print("Polling updates...")
                updates = await bot.get_updates(offset=last_update_id + 1, timeout=30)
                for update in updates:
                    if update.message and update.message.text:
                        if is_authorized_user(update):
                            response_queue = update.message.text
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
