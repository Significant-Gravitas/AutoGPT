import asyncio
import json
import random

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.utils import executor

import commands as cmd
import utils
from memory import get_memory
import data
import chat
from spinner import Spinner
import time
import speak
from config import Config
from json_parser import fix_and_parse_json
from ai_config import AIConfig
import traceback
import yaml
import argparse
import logging

cfg = Config()
ai_config = AIConfig()
# setup telegram bot
bot = Bot(token=cfg.telegram_bot_token)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
cfg.bot = bot


class Form(StatesGroup):
    name = State()
    role = State()
    goals = State()
    confirm = State()


def configure_logging():
    logging.basicConfig(filename='log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    return logging.getLogger('AutoGPT')

def check_openai_api_key():
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    if not cfg.openai_api_key:
        print(
     +
            "Please set your OpenAI API key in config.py or as an environment variable."
        )
        print("You can get your key from https://beta.openai.com/account/api-keys")
        exit(1)


async def print_to_console(
        title,
        content,
        speak_text=False,
        min_typing_speed=0.05,
        max_typing_speed=0.01):
    """Prints text to the console with a typing effect"""
    global cfg
    global logger
    await cfg.bot.send_message(cfg.chat_id, title + ": " + content)


async def attempt_to_fix_json_by_finding_outermost_brackets(json_string):
    if cfg.speak_mode and cfg.debug_mode:
      speak.say_text("I have received an invalid JSON response from the OpenAI API. Trying to fix it now.")
    # print_to_console("Attempting to fix JSON by finding outermost brackets\n", "")

    try:
        # Use regex to search for JSON objects
        import regex
        json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
        json_match = json_pattern.search(json_string)

        if json_match:
            # Extract the valid JSON object from the string
            json_string = json_match.group(0)
            # await print_to_console("Apparently json was fixed.","")
            if cfg.speak_mode and cfg.debug_mode:
               speak.say_text("Apparently json was fixed.")
        else:
            raise ValueError("No valid JSON object found")

    except (json.JSONDecodeError, ValueError) as e:
        if cfg.speak_mode:
            speak.say_text("Didn't work. I will have to ignore this response then.")
        # await print_to_console("Error: Invalid JSON, setting it to empty JSON now.\n", "")
        json_string = {}

    return json_string


async def print_assistant_thoughts(assistant_reply):
    """Prints the assistant's thoughts to the console"""
    global cfg
    try:
        try:
            # Parse and print Assistant response
            assistant_reply_json = fix_and_parse_json(assistant_reply)
        except json.JSONDecodeError as e:
            await print_to_console("Error: Invalid JSON in assistant thoughts\n", assistant_reply)
            assistant_reply_json = await attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply)
            assistant_reply_json = fix_and_parse_json(assistant_reply_json)

        # Check if assistant_reply_json is a string and attempt to parse it into a JSON object
        if isinstance(assistant_reply_json, str):
            try:
                assistant_reply_json = json.loads(assistant_reply_json)
            except json.JSONDecodeError as e:
                await print_to_console("Error: Invalid JSON in assistant thoughts\n", assistant_reply)
                assistant_reply_json = await attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply_json)

        assistant_thoughts_reasoning = None
        assistant_thoughts_plan = None
        assistant_thoughts_speak = None
        assistant_thoughts_criticism = None
        assistant_thoughts = assistant_reply_json.get("thoughts", {})
        assistant_thoughts_text = assistant_thoughts.get("text")

        if assistant_thoughts:
            assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
            assistant_thoughts_plan = assistant_thoughts.get("plan")
            assistant_thoughts_criticism = assistant_thoughts.get("criticism")
            assistant_thoughts_speak = assistant_thoughts.get("speak")

        await print_to_console(f"{ai_config.ai_name.upper()} THOUGHTS:", assistant_thoughts_text)
        await print_to_console("REASONING:", assistant_thoughts_reasoning)

        if assistant_thoughts_plan:
            await print_to_console("PLAN:", "")
            # If it's a list, join it into a string
            if isinstance(assistant_thoughts_plan, list):
                assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
            elif isinstance(assistant_thoughts_plan, dict):
                assistant_thoughts_plan = str(assistant_thoughts_plan)

            # Split the input_string using the newline character and dashes
            lines = assistant_thoughts_plan.split('\n')
            for line in lines:
                line = line.lstrip("- ")
                await print_to_console("- ", line.strip())

        await print_to_console("CRITICISM:", assistant_thoughts_criticism)
        # Speak the assistant's thoughts
        if cfg.speak_mode and assistant_thoughts_speak:
            speak.say_text(assistant_thoughts_speak)
        
        return assistant_reply_json
    except json.decoder.JSONDecodeError as e:
        await print_to_console("Error: Invalid JSON\n", assistant_reply)
        if cfg.speak_mode:
            speak.say_text("I have received an invalid JSON response from the OpenAI API. I cannot ignore this response.")

    # All other errors, return "Error: + error message"
    except Exception as e:
        call_stack = traceback.format_exc()
        await print_to_console("Error: \n", call_stack)


def load_variables(config_file="config.yaml"):
    """Load variables from yaml file if it exists, otherwise prompt the user for input"""
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        ai_name = config.get("ai_name")
        ai_role = config.get("ai_role")
        ai_goals = config.get("ai_goals")
    except FileNotFoundError:
        ai_name = ""
        ai_role = ""
        ai_goals = []

    # Prompt the user for input if config file is missing or empty values
    if not ai_name:
        ai_name = utils.clean_input("Name your AI: ")
        if ai_name == "":
            ai_name = "Entrepreneur-GPT"

    if not ai_role:
        ai_role = utils.clean_input(f"{ai_name} is: ")
        if ai_role == "":
            ai_role = "an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth."

    if not ai_goals:
        print("Enter up to 5 goals for your AI: ")
        print("For example: \nIncrease net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'")
        print("Enter nothing to load defaults, enter nothing when finished.")
        ai_goals = []
        for i in range(5):
            ai_goal = utils.clean_input(f"Goal {i+1}: ")
            if ai_goal == "":
                break
            ai_goals.append(ai_goal)
        if len(ai_goals) == 0:
            ai_goals = ["Increase net worth", "Grow Twitter Account", "Develop and manage multiple businesses autonomously"]

    # Save variables to yaml file
    config = {"ai_name": ai_name, "ai_role": ai_role, "ai_goals": ai_goals}
    with open(config_file, "w") as file:
        documents = yaml.dump(config, file)

    prompt = data.load_prompt()
    prompt_start = """Your decisions must always be made independently without seeking user assistance. Play to your strengths as a LLM and pursue simple strategies with no legal complications."""

    # Construct full prompt
    full_prompt = f"You are {ai_name}, {ai_role}\n{prompt_start}\n\nGOALS:\n\n"
    for i, goal in enumerate(ai_goals):
        full_prompt += f"{i+1}. {goal}\n"

    full_prompt += f"\n\n{prompt}"
    return full_prompt


async def construct_prompt():
    """Construct the prompt for the AI to respond to"""
    full_prompt = ai_config.construct_full_prompt()
    return full_prompt


@dp.message_handler(state=Form.name)
async def set_ai_name(message: types.Message, state: FSMContext):
    """Set the AI's name"""
    async with state.proxy() as data:
        data['name'] = message.text
    await Form.next()
    await message.reply(
        f'{message.text} here! Now, describe your AI\'s role.'
    )


@dp.message_handler(state=Form.role)
async def set_ai_role(message: types.Message, state: FSMContext):
    """Set the AI's role"""
    async with state.proxy() as data:
        data['role'] = message.text
    await Form.next()
    await message.reply(
        f'Now, enter up to 5 goals for your AI. Enter nothing to load defaults, enter nothing when finished.'
    )


@dp.message_handler(state=Form.goals)
async def set_ai_goals(message: types.Message, state: FSMContext):
    """Set the AI's goals"""
    async with state.proxy() as data:
        data['goals'] = message.text.split(';')
    await Form.next()
    await message.reply(
        f'Your AI\'s name is {data["name"]}, its role is {data["role"]}, and its goals are {data["goals"]}.'
        f' Would you like to continue with these settings? (y/n)'
    )


@dp.message_handler(state=Form.confirm)
async def confirm_settings(message: types.Message, state: FSMContext):
    """Confirm the AI's settings"""
    global ai_config
    async with state.proxy() as data:
        data['confirm'] = message.text
    if data['confirm'].lower() == "y":
        ai_config = AIConfig(
            ai_name=data['name'],
            ai_role=data['role'],
            ai_goals=data['goals'],
        )
        await state.finish()
        await message.reply(
            f'Your AI\'s name is {data["name"]}, its role is {data["role"]}, and its goals are {data["goals"]}.'
            f' Okkkaayyy, let\'s goooo'
        )
    await do_magic()


@dp.message_handler(commands=['start'])
async def prompt_user(message: types.Message):
    """Prompt the user for input"""
    # Construct the prompt
    cfg.chat_id = message.chat.id
    await Form.name.set()
    # Get AI Name from User
    await message.reply("Welcome to Auto-GPT! Enter the name of your AI and its role below.")


async def parse_arguments():
    """Parses the arguments passed to the script"""
    global cfg
    cfg.set_debug_mode(False)
    cfg.set_continuous_mode(True)
    cfg.set_speak_mode(False)

    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--continuous', action='store_true', help='Enable Continuous Mode')
    parser.add_argument('--speak', action='store_true', help='Enable Speak Mode')
    parser.add_argument('--debug', action='store_true', help='Enable Debug Mode')
    parser.add_argument('--gpt3only', action='store_true', help='Enable GPT3.5 Only Mode')
    parser.add_argument('--gpt4only', action='store_true', help='Enable GPT4 Only Mode')
    args = parser.parse_args()

    if args.continuous:
        await print_to_console("Continuous Mode: ", "ENABLED")
        await print_to_console(
            "WARNING: ",
            "Continuous mode is not recommended. It is potentially dangerous and may cause your AI to ru carry out actions you would not usually authorise. Use at your own risk.")
        cfg.set_continuous_mode(True)

    if args.speak:
        await print_to_console("Speak Mode: ", "ENABLED")
        cfg.set_speak_mode(False)

    if args.gpt3only:
        await print_to_console("GPT3.5 Only Mode: ", "ENABLED")
        cfg.set_smart_llm_model(cfg.fast_llm_model)
    
    if args.gpt4only:
        await print_to_console("GPT4 Only Mode: ", "ENABLED")
        cfg.set_fast_llm_model(cfg.smart_llm_model)

    if args.debug:
        await print_to_console("Debug Mode: ", "ENABLED")
        cfg.set_debug_mode(True)


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    """Sends a help message to the user"""
    await message.answer("Here is a list of commands:\n\n"
                         "/start - Start the AI\n"
                         "/stop - Stop the AI\n")


async def do_magic():
    cfg = Config()
    check_openai_api_key()
    await parse_arguments()
    prompt = await construct_prompt()
    memory = get_memory(cfg, init=True)
    full_message_history = []
    next_action_count = 0
    # Make a constant:
    user_input = "Determine which next command to use, and respond using the format specified above:"
    run = True
    command_name = None

    @dp.message_handler(commands=['stop'])
    async def stop(message: types.Message):
        """Stops the AI"""
        nonlocal run
        await message.answer('Stopping AI...')
        await asyncio.sleep(1)
        run = False
        await message.answer('AI stopped.')

    # Interaction Loop
    while run:
        # Send message to AI, get response
        with Spinner("Thinking... "):
            assistant_reply = await chat.chat_with_ai(
                prompt,
                user_input,
                full_message_history,
                memory,
                cfg.fast_token_limit) # TODO: This hardcodes the model to use GPT3.5. Make this an argument

        # Print Assistant thoughts
        await print_assistant_thoughts(assistant_reply)

        # Get command name and arguments
        try:
            command_name, arguments = cmd.get_command(await attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply))
            if cfg.speak_mode:
                speak.say_text(f"I want to execute {command_name}")
        except Exception as e:
            await print_to_console("Error: \n", str(e))

        # Print command
        await print_to_console(
            "NEXT ACTION: ",
            f"COMMAND = {command_name}  ARGUMENTS = {arguments}")

        # Execute command
        if command_name is not None and command_name.lower().startswith("error"):
            result = f"Command {command_name} threw the following error: " + arguments
        elif command_name == "human_feedback":
            result = f"Human feedback: {user_input}"
        else:
            result = f"Command {command_name} returned: {cmd.execute_command(command_name, arguments)}"
            if next_action_count > 0:
                next_action_count -= 1
        if result == 'stop':
            break
        memory_to_add = f"Assistant Reply: {assistant_reply} " \
                        f"\nResult: {result} " \
                        f"\nHuman Feedback: {user_input} "

        memory.add(memory_to_add)

        # Check if there's a result from the command append it to the message
        # history
        if result is not None:
            full_message_history.append(chat.create_chat_message("system", result))
            await print_to_console("SYSTEM: ", result)
        else:
            full_message_history.append(
                chat.create_chat_message(
                    "system", "Unable to execute command"))
            await print_to_console("SYSTEM: ", "Unable to execute command")

if __name__ == '__main__':
    logger = configure_logging()
    executor.start_polling(dp)
