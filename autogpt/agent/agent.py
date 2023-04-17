from colorama import Fore, Style
from autogpt.app import execute_command, get_command

from autogpt.chat import chat_with_ai, create_chat_message
from autogpt.config import Config
from autogpt.json_fixes.master_json_fix_method import fix_json_using_multiple_techniques
from autogpt.json_validation.validate_json import validate_json
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input


class Agent:
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_name: The name of the agent.
        memory: The memory object to use.
        full_message_history: The full message history.
        next_action_count: The number of actions to execute.
        system_prompt: The system prompt is the initial prompt that defines everything the AI needs to know to achieve its task successfully.
        Currently, the dynamic and customizable information in the system prompt are ai_name, description and goals.

        triggering_prompt: The last sentence the AI will see before answering. For Auto-GPT, this prompt is:
            Determine which next command to use, and respond using the format specified above:
            The triggering prompt is not part of the system prompt because between the system prompt and the triggering
            prompt we have contextual information that can distract the AI and make it forget that its goal is to find the next task to achieve.
            SYSTEM PROMPT
            CONTEXTUAL INFORMATION (memory, previous conversations, anything relevant)
            TRIGGERING PROMPT

        The triggering prompt reminds the AI about its short term meta task (defining the next task)
    """

    def __init__(
        self,
        ai_name,
        memory,
        full_message_history,
        next_action_count,
        system_prompt,
        triggering_prompt,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt

    def start_interaction_loop(self):
        # Interaction Loop
        cfg = Config()
        loop_count = 0
        command_name = None
        arguments = None
        user_input = ""

        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1
            if (
                cfg.continuous_mode
                and cfg.continuous_limit > 0
                and loop_count > cfg.continuous_limit
            ):
                logger.typewriter_log(
                    "Continuous Limit Reached: ", Fore.YELLOW, f"{cfg.continuous_limit}"
                )
                break

            # Send message to AI, get response
            with Spinner("Thinking... "):
                assistant_reply = chat_with_ai(
                    self.system_prompt,
                    self.triggering_prompt,
                    self.full_message_history,
                    self.memory,
                    cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            assistant_reply_json = fix_json_using_multiple_techniques(assistant_reply)

            # Print Assistant thoughts
            if assistant_reply_json != {}:
                validate_json(assistant_reply_json, 'llm_response_format_1')
                # Get command name and arguments
                try:
                    print_assistant_thoughts(self.ai_name, assistant_reply_json)
                    command_name, arguments = get_command(assistant_reply_json)
                    # command_name, arguments = assistant_reply_json_valid["command"]["name"], assistant_reply_json_valid["command"]["args"]
                    if cfg.speak_mode:
                        say_text(f"I want to execute {command_name}")
                except Exception as e:
                    logger.error("Error: \n", str(e))

            if not cfg.continuous_mode and self.next_action_count == 0:
                ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                    f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
                )
                print(
                    "Enter 'y' to authorise command, 'y -N' to run N continuous "
                    "commands, 'n' to exit program, or enter feedback for "
                    f"{self.ai_name}...",
                    flush=True,
                )
                while True:
                    console_input = clean_input(
                        Fore.MAGENTA + "Input:" + Style.RESET_ALL
                    )
                    if console_input.lower().rstrip() == "y":
                        user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().startswith("y -"):
                        try:
                            self.next_action_count = abs(
                                int(console_input.split(" ")[1])
                            )
                            user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            print(
                                "Invalid input format. Please enter 'y -n' where n is"
                                " the number of continuous tasks."
                            )
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
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                        Fore.MAGENTA,
                        "",
                    )
                elif user_input == "EXIT":
                    print("Exiting...", flush=True)
                    break
            else:
                # Print command
                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}"
                    f"  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
                )

            # Execute command
            if command_name is not None and command_name.lower().startswith("error"):
                result = (
                    f"Command {command_name} threw the following error: {arguments}"
                )
            elif command_name == "human_feedback":
                result = f"Human feedback: {user_input}"
            else:
                result = (
                    f"Command {command_name} returned: "
                    f"{execute_command(command_name, arguments)}"
                )
                if self.next_action_count > 0:
                    self.next_action_count -= 1

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} "
                f"\nResult: {result} "
                f"\nHuman Feedback: {user_input} "
            )

            self.memory.add(memory_to_add)

            # Check if there's a result from the command append it to the message
            # history
            if result is not None:
                self.full_message_history.append(create_chat_message("system", result))
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            else:
                self.full_message_history.append(
                    create_chat_message("system", "Unable to execute command")
                )
                logger.typewriter_log(
                    "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
                )
