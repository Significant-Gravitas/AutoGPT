import argparse
import chat
from config import Config
from colorama import Fore, Style
import commands as cmd
from memory import get_memory
from spinner import Spinner
from run_utils import (
    print_to_console,
    get_ai_config,
    print_assistant_thoughts,
)


DEF_USER_INPUT = (
    "Determine which next command to use, and respond using the format specified above:"
)


class AutoGPT:
    def __init__(
        self,
        continous_mode=False,
        speak_mode=False,
        gpt3only_mode=False,
    ):
        self.cfg = Config()
        if continous_mode:
            print_to_console("Continuous Mode: ", Fore.RED, "ENABLED")
            print_to_console(
                "WARNING: ",
                Fore.RED,
                "Continuous mode is not recommended. It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorise. Use at your own risk.",
            )
            self.cfg.set_continuous_mode(True)
        if speak_mode:
            print_to_console("Speak Mode: ", Fore.GREEN, "ENABLED")
            self.cfg.set_speak_mode(True)
        if gpt3only_mode:
            print_to_console("GPT3.5 Only Mode: ", Fore.GREEN, "ENABLED")
            self.cfg.set_smart_llm_model(self.cfg.fast_llm_model)
        self.ai_config = get_ai_config(self.cfg.speak_mode)
        self.prompt = self.ai_config.construct_full_prompt()
        self.memory = get_memory(self.cfg, init=True)
        self.full_message_history = []
        print("Using memory of type: " + self.memory.__class__.__name__)

    def run(self):
        next_action_count = 0
        while True:
            # Send message to AI, get response
            with Spinner("Thinking... "):
                assistant_reply = chat.chat_with_ai(
                    self.prompt,
                    DEF_USER_INPUT,
                    self.full_message_history,
                    self.memory,
                    self.cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument
            print_assistant_thoughts(
                self.ai_config.ai_name, assistant_reply, self.cfg.speak_mode
            )

            # Get command name and arguments
            command_name = ""
            arguments = ""

            command_name, arguments = cmd.get_command(assistant_reply)

            if command_name == "Error":
                print_to_console("Failed: \n", Fore.RED, f"ERROR = {arguments}")

            print_to_console(
                "NEXT ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
            )
            user_input = ""
            if not self.cfg.continuous_mode and next_action_count == 0:
                # User authorization: Prompt the user to press y/n to continue or leave.
                ask_user_input = True
                while ask_user_input:
                    print(
                        f"Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for {self.ai_config.ai_name}...",
                        flush=True,
                    )
                    ask_user_input = False
                    console_input = input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)

                    if console_input.lower().startswith("y"):
                        user_input = "GENERATE NEXT COMMAND JSON"
                        if console_input.lower() == "y":
                            message = "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-="
                        else:
                            try:
                                next_action_count = abs(
                                    int(console_input.split(" ")[1])
                                )
                                message = f"-=-=-=-=-=-=-= NEXT {next_action_count} COMMAND AUTHORISED BY USER -=-=-=-=-=-=-="
                            except ValueError:
                                print(
                                    "Invalid input format. Please enter 'y -n' where n is the number of continuous tasks."
                                )
                                ask_user_input = True  # Ask user for input again
                                continue
                        print_to_console(
                            message,
                            Fore.MAGENTA,
                            "",
                        )
                    elif console_input.lower() == "n":
                        print("Exiting...", flush=True)
                        return
                    else:
                        user_input = console_input
                        command_name = "human_feedback"

            # Execute command
            if command_name.lower() == "error":
                result = f"Command {command_name} threw the following error: " + str(
                    arguments
                )
            elif command_name == "human_feedback":
                result = f"Human feedback: {user_input}"
            else:
                result = f"Command {command_name} returned: {cmd.execute_command(command_name, arguments)}"
                if next_action_count > 0:
                    next_action_count -= 1

            # Save assistant reply and result to memory
            self.memory.add(
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )

            # Append result to the message history
            self.full_message_history.append(chat.create_chat_message("system", result))
            print_to_console("SYSTEM: ", Fore.YELLOW, result)


def main():
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument(
        "--continuous", action="store_true", help="Enable Continuous Mode"
    )
    parser.add_argument("--speak", action="store_true", help="Enable Speak Mode")
    parser.add_argument("--debug", action="store_true", help="Enable Debug Mode")
    parser.add_argument(
        "--gpt3only", action="store_true", help="Enable GPT3.5 Only Mode"
    )
    args = parser.parse_args()
    # Initialize AutoGPT
    auto_gpt = AutoGPT(
        continous_mode=args.continuous,
        speak_mode=args.speak,
        gpt3only_mode=args.gpt3only,
    )
    auto_gpt.run()


if __name__ == "__main__":
    main()
