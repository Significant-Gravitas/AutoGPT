import asyncio
import os
import traceback

import requests
import speech_recognition as sr
import yaml
from colorama import Fore
from git import Repo

import autogpt.speech.say as speak
from autogpt.config.config import Config
from autogpt.telegram_chat import TelegramUtils

cfg = Config()


def clean_input(prompt: str = "", talk=False):
    try:
        if talk and cfg.use_mac_os_voice_input == "True":
            try:
                return voice_input(prompt)
            except:
                print(traceback.format_exc())
                print("Siri could not understand your input.")
                speak.say_text("I didn't understand that. Sorry.")
                return input(prompt)
        else:
            if cfg.telegram_enabled:
                print("Asking user via Telegram...")
                telegramUtils = TelegramUtils()
                chat_answer = telegramUtils.ask_user(prompt=prompt)
                print("Telegram answer: " + chat_answer)
                if chat_answer in [
                    "yes",
                    "yeah",
                    "yep",
                    "yup",
                    "y",
                    "ok",
                    "okay",
                    "sure",
                    "affirmative",
                    "aye",
                    "aye aye",
                    "alright",
                    "alrighty",
                ]:
                    return "y"
                elif chat_answer in [
                    "no",
                    "nope",
                    "n",
                    "nah",
                    "negative",
                    "nay",
                    "nay nay",
                ]:
                    return "n"
                return chat_answer

            # ask for input, default when just pressing Enter is y
            print("Asking user via keyboard...")
            answer = input(prompt + " [y/n] or press Enter for default (y): ")
            if answer == "":
                answer = "y"
            return answer
    except KeyboardInterrupt:
        print("You interrupted Auto-GPT from utils.py")
        print("Quitting...")
        exit(0)


def voice_input(prompt: str = "", voice_prompt_counter: int = 0):
    recognizer = sr.Recognizer()

    if voice_prompt_counter > 3:
        speak.macos_tts_speech(
            "I'm sorry, I didn't understand that. Please use the keyboard."
        )
        return clean_input(prompt, talk=False)

    voice_prompt_counter += 1
    try:
        speak.macos_tts_speech(prompt)
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        try:
            user_input = recognizer.recognize_sphinx(audio)

            if user_input in [
                "yes",
                "yeah",
                "yep",
                "yup",
                "y",
                "ok",
                "okay",
                "sure",
                "affirmative",
                "aye",
                "aye aye",
                "alright",
                "alrighty",
            ]:
                return "y"
            elif user_input in ["no", "nope", "n", "nah", "negative", "nay", "nay nay"]:
                return "n"
            elif user_input in [
                "quit",
                "exit",
                "stop",
                "end",
                "terminate",
                "cancel",
                "break",
                "halt",
                "die",
                "kill",
                "terminate",
            ]:
                speak.macos_tts_speech("Okay then.. Goodbye!")
                print("You interrupted Auto-GPT")
                print("Quitting...")
                exit(0)
            elif user_input in [
                "do what you want",
                "keep going",
                "keep on going",
                "keep on",
            ]:
                return "y -15"
            else:
                print("You said: " + user_input)
                return input(prompt)

        except sr.UnknownValueError:
            print(traceback.format_exc())
            print("Siri could not understand your input.")
            speak.macos_tts_speech("I didn't understand that. Sorry.")
            return input(prompt)
    except KeyboardInterrupt:
        print("You interrupted Auto-GPT")
        print("Quitting...")
        exit(0)


def validate_yaml_file(file: str):
    try:
        with open(file, encoding="utf-8") as fp:
            yaml.load(fp.read(), Loader=yaml.FullLoader)
    except FileNotFoundError:
        return (False, f"The file {Fore.CYAN}`{file}`{Fore.RESET} wasn't found")
    except yaml.YAMLError as e:
        return (
            False,
            f"There was an issue while trying to read with your AI Settings file: {e}",
        )

    return (True, f"Successfully validated {Fore.CYAN}`{file}`{Fore.RESET}!")


def readable_file_size(size, decimal_places=2):
    """Converts the given size in bytes to a readable format.
    Args:
        size: Size in bytes
        decimal_places (int): Number of decimal places to display
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def get_bulletin_from_web() -> str:
    try:
        response = requests.get(
            "https://raw.githubusercontent.com/Significant-Gravitas/Auto-GPT/master/BULLETIN.md"
        )
        if response.status_code == 200:
            return response.text
    except:
        return ""


def get_current_git_branch() -> str:
    try:
        repo = Repo(search_parent_directories=True)
        branch = repo.active_branch
        return branch.name
    except:
        return ""


def get_latest_bulletin() -> str:
    exists = os.path.exists("CURRENT_BULLETIN.md")
    current_bulletin = ""
    if exists:
        current_bulletin = open("CURRENT_BULLETIN.md", "r", encoding="utf-8").read()
    new_bulletin = get_bulletin_from_web()
    is_new_news = new_bulletin != current_bulletin

    if new_bulletin and is_new_news:
        open("CURRENT_BULLETIN.md", "w", encoding="utf-8").write(new_bulletin)
        return f" {Fore.RED}::UPDATED:: {Fore.CYAN}{new_bulletin}{Fore.RESET}"
    return current_bulletin
