import sys
from enum import Enum, auto

from colorama import Fore, Style

import chat
import commands as cmd
import data
from ai_config import AIConfig
from argument_parser import parse_arguments
from config import Config
from console_interaction import print_assistant_thoughts, print_to_console, prompt_user
from memory import get_memory
from spinner import Spinner
import yaml


def main_interaction_loop(config, prompt):
    cfg = Config()
    parse_arguments()

    # Initialize variables
    full_message_history = []
    result = None
    next_action_count = 0
    user_input = "Determine which next command to use, and respond using the format specified above:"
    
    # Initialize memory and make sure it is empty.
    memory = get_memory(cfg, init=True)
    print('Using memory of type: ' + memory.__class__.__name__)
    
    # Interaction Loop
    while True:
        # Send message to AI, get response
        with Spinner("Thinking... "):
            assistant_reply = chat.chat_with_ai(
                prompt,
                user_input,
                full_message_history,
                memory,
                cfg.fast_token_limit, cfg.debug)

        # Print Assistant thoughts
        print_assistant_thoughts(config.ai_name, assistant_reply)

        # Get command name and arguments
        try:
            command_name, arguments = cmd.get_command(assistant_reply)
        except Exception as e:
            print_to_console("Error: \n", Fore.RED, str(e))

        if not cfg.continuous_mode and next_action_count == 0:
            ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
            user_input = ""
            print_to_console(
                "NEXT ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")
            print(
                f"Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for {config.ai_name}...",
                flush=True)
            while True:
                console_input = input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)
                if console_input.lower() == "y":
                    user_input = "GENERATE NEXT COMMAND JSON"
                    break
                elif console_input.lower().startswith("y -"):
                    try:
                        next_action_count = abs(int(console_input.split(" ")[1]))
                        user_input = "GENERATE NEXT COMMAND JSON"
                    except ValueError:
                        print("Invalid input format. Please enter 'y -n' where n is the number of continuous tasks.")
                        continue
                    break
                elif console_input.lower() == "n":
                    user_input = "EXIT"
                    break
                else:
                    user_input = console_input
                    command_name = "human_feedback"
                    break

            if user_input == "GENERATE NEXT COMMAND JSON":
                print_to_console(
                "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                Fore.MAGENTA,
                "")
            elif user_input == "EXIT":
                print("Exiting...", flush=True)
                break
        else:
            # Print command
            print_to_console(
                "NEXT ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")

        # Execute command
        if command_name.lower().startswith("error"):
            result = f"Command {command_name} threw the following error: " + arguments
        elif command_name == "human_feedback":
            result = f"Human feedback: {user_input}"
        else:
            result = f"Command {command_name} returned: {cmd.execute_command(command_name, arguments)}"
            if next_action_count > 0:
                next_action_count -= 1

        memory_to_add = f"Assistant Reply: {assistant_reply} " \
                        f"\nResult: {result} " \
                        f"\nHuman Feedback: {user_input} "

        memory.add(memory_to_add)

        # Check if there's a result from the command append it to the message
        # history
        if result is not None:
            full_message_history.append(chat.create_chat_message("system", result))
            print_to_console("SYSTEM: ", Fore.YELLOW, result)
        else:
            full_message_history.append(
                chat.create_chat_message(
                    "system", "Unable to execute command"))
            print_to_console("SYSTEM: ", Fore.YELLOW, "Unable to execute command")


if __name__ == "__main__":
    config = AIConfig.load()
    if config.ai_name:
        print_to_console(
            f"Welcome back! ",
            Fore.GREEN,
            f"Would you like me to return to being {config.ai_name}?",
            speak_text=True)
        should_continue = input(f"""Continue with the last settings? 
    Name:  {config.ai_name}
    Role:  {config.ai_role}
    Goals: {config.ai_goals}  
    Continue (y/n): """)
        if should_continue.lower() == "n":
            config = AIConfig()

    if not config.ai_name:         
        config = prompt_user()
        config.save()

    prompt = config.construct_full_prompt()
    main_interaction_loop(config, prompt)