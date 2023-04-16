from colorama import Fore, Style

from autogpt.app import get_command, execute_command
from autogpt.chat import chat_with_ai, create_chat_message
from autogpt.config import Singleton
from autogpt.json_fixes.bracket_termination import attempt_to_fix_json_by_finding_outermost_brackets
from autogpt.logs import logger, print_assistant_thoughts
from multigpt.memory import get_memory
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from multigpt.multi_agent import MultiAgent
from multigpt.expert import Expert


class MultiAgentManager(metaclass=Singleton):

    def __init__(self, cfg):
        self.cfg = cfg
        self.agents = []
        self.agent_counter = 0
        self.next_action_count = 0

    def create_agent(self, expert: Expert):
        user_input = (
            "Determine which next command to use, and respond using the"
            " format specified above:"
        )
        prompt = expert.construct_full_prompt()

        agent_id = self.agent_counter
        self.agent_counter += 1

        memory = get_memory(self.cfg, ai_key=agent_id, init=True)
        logger.typewriter_log(
            f"Using memory of type:", Fore.GREEN, f"{memory.__class__.__name__}"
        )

        agent = MultiAgent(
            ai_name=expert.ai_name,
            memory=memory,
            full_message_history=[],
            prompt=prompt,
            user_input=user_input,
            agent_id=agent_id
        )
        self.agents.append(agent)

    def get_active_agent(self, loop_count):
        return self.agents[loop_count % len(self.agents)]

    def start_interaction_loop(self):
        # Interaction Loop
        loop_count = 0
        command_name = None
        arguments = None
        while True:
            # Discontinue if continuous limit is reached
            loop_count += 1
            if (
                    self.cfg.continuous_mode
                    and self.cfg.continuous_limit > 0
                    and loop_count > self.cfg.continuous_limit
            ):
                logger.typewriter_log(
                    "Continuous Limit Reached: ", Fore.YELLOW, f"{self.cfg.continuous_limit}"
                )
                break
            active_agent = self.get_active_agent(loop_count)

            # Send message to AI, get response
            with Spinner("Thinking... "):
                assistant_reply = chat_with_ai(
                    active_agent.prompt,
                    active_agent.user_input,
                    active_agent.full_message_history,
                    active_agent.memory,
                    self.cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            # Print Assistant thoughts
            print_assistant_thoughts(active_agent.ai_name, assistant_reply)

            # Get command name and arguments
            try:
                command_name, arguments = get_command(
                    attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply)
                )
                if self.cfg.speak_mode:
                    say_text(f"I want to execute {command_name}")
            except Exception as e:
                logger.error("Error: \n", str(e))

            if not self.cfg.continuous_mode and self.next_action_count == 0:
                ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                active_agent.user_input = ""
                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                    f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
                )
                print(
                    "Enter 'y' to authorise command, 'y -N' to run N continuous "
                    "commands, 'n' to exit program, or enter feedback for "
                    f"{active_agent.ai_name}...",
                    flush=True,
                )
                while True:
                    console_input = clean_input(
                        Fore.MAGENTA + "Input:" + Style.RESET_ALL
                    )
                    if console_input.lower().rstrip() == "y":
                        active_agent.user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().startswith("y -"):
                        try:
                            self.next_action_count = abs(
                                int(console_input.split(" ")[1])
                            )
                            active_agent.user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            print(
                                "Invalid input format. Please enter 'y -n' where n is"
                                " the number of continuous tasks."
                            )
                            continue
                        break
                    elif console_input.lower() == "n":
                        active_agent.user_input = "EXIT"
                        break
                    else:
                        active_agent.user_input = console_input
                        command_name = "human_feedback"
                        break

                if active_agent.user_input == "GENERATE NEXT COMMAND JSON":
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                        Fore.MAGENTA,
                        "",
                    )
                elif active_agent.user_input == "EXIT":
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
                result = f"Human feedback: {active_agent.user_input}"
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
                f"\nHuman Feedback: {active_agent.user_input} "
            )

            active_agent.memory.add(memory_to_add)

            # Check if there's a result from the command append it to the message
            # history
            if result is not None:
                active_agent.full_message_history.append(create_chat_message("system", result))
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            else:
                active_agent.full_message_history.append(
                    create_chat_message("system", "Unable to execute command")
                )
                logger.typewriter_log(
                    "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
                )
