import asyncio
import os
import random
import re
from enum import Enum

from colorama import Fore, Style
from slugify import slugify

from autogpt.app import get_command, execute_command
from autogpt.chat import chat_with_ai, create_chat_message
from autogpt.config import Singleton
from autogpt.json_fixes.bracket_termination import attempt_to_fix_json_by_finding_outermost_brackets
from autogpt.llm_utils import create_chat_completion
from autogpt.logs import logger, print_assistant_thoughts
from multigpt.memory import get_memory
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from multigpt.multi_agent import MultiAgent
from multigpt.agent_selection import AgentSelection
from multigpt.constants import CHAT_ONLY_MODE, NEXTAGENTSELECTION, USE_LMQL_QUERIES
from multigpt import lmql_utils


class MultiAgentManager(metaclass=Singleton):

    def __init__(self, cfg):
        self.cfg = cfg
        self.agents = []
        self.agent_counter = 0
        self.next_action_count = 0
        self.experts = []
        self.last_active_agent = None
        self.current_active_agent = None
        self.chat_buffer = []
        self.chat_buffer_size = 10

    def set_experts(self, experts):
        self.experts = experts

    def create_agent(self, expert):
        slugified_filename = slugify(expert.ai_name, separator="_", lowercase=True) + "_settings.yaml"

        saved_agents_directory = os.path.join(os.path.dirname(__file__), "saved_agents")
        if not os.path.exists(saved_agents_directory):
            print(
                "saved_agents directory does not exist yet."
                f"Creating saved_agents..."
            )
            os.mkdir(saved_agents_directory)
        # TODO: sometimes doesn't find the file, because it hasn't finished saving yet
        filepath = os.path.join(saved_agents_directory, f"{slugified_filename}")
        expert.save(filepath)

        user_input = (
            "Determine which next command to use, and respond using the"
            " format specified above:"
        )
        prompt = expert.construct_full_prompt()

        agent_id = self.agent_counter
        self.agent_counter += 1

        memory = get_memory(self.cfg, ai_key=agent_id, init=True)
        if not CHAT_ONLY_MODE:
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

    def chat_buffer_to_str(self):
        res = ""
        for item in self.chat_buffer:
            agent, message = item
            res += f"{agent.ai_name}: {message}\n"
        return res

    def agents_to_str(self):
        res = ""
        for i, agent in enumerate(self.agents, start=1):
            res += f"{i} - {agent.ai_name}\n"
        return res

    def parse_num_output_llm(self, input_string):
        pattern = r'\d+'

        match = re.search(pattern, input_string)

        if match:
            return int(match.group())
        return None

    def get_active_agent(self, loop_count):
        self.last_active_agent = self.current_active_agent
        self.current_active_agent = None

        if NEXTAGENTSELECTION == AgentSelection.ROUND_ROBIN:
            self.current_active_agent = self.agents[loop_count % len(self.agents)]

        elif NEXTAGENTSELECTION == AgentSelection.RANDOM:
            self.current_active_agent = self.agents[random.randint(0, len(self.agents) - 1)]

        elif NEXTAGENTSELECTION == AgentSelection.SMART_SELECTION:
            if self.last_active_agent is None and len(
                    self.agents) > 0:  # If last agent is None, fallback to random select
                self.current_active_agent = self.agents[random.randint(0, len(self.agents) - 1)]
            else:
                try:
                    with Spinner("Selecting next participant... "):
                        next_speaker_id, _, reasoning = lmql_utils.lmql_smart_select(self.chat_buffer_to_str(), self.agents_to_str())
                        print(f"\n\n{reasoning}\n\n")
                        # If smart select selects same agent, use random select instead
                        if self.last_active_agent is self.agents[next_speaker_id - 1]:
                            self.current_active_agent = self.agents[random.randint(0, len(self.agents) - 1)]
                        else:
                            self.current_active_agent = self.agents[next_speaker_id - 1]
                except Exception as e:  # If smart select fails for some reason, just fallback to random select
                    self.current_active_agent = self.agents[random.randint(0, len(self.agents) - 1)]

        else:
            raise ValueError("Invalid agent selection. Only use appropriate values for const SELECTION.")
        return self.current_active_agent

    def send_message_to_all_agents(self, speaker=None, message=None):
        def message_is_empty():
            if message is None:
                return True
            pattern = re.compile(r'(\d|[a-z]|[A-Z])')
            return not bool(pattern.match(message)) or len(message) == 0

        if speaker is None or message_is_empty():
            return False
        for agent in self.agents:
            agent.send_message(speaker, message)
        self.add_message_to_chat_buffer(speaker, message)
        return True

    def add_message_to_chat_buffer(self, speaker, message):
        self.chat_buffer.append((speaker, message))
        if len(self.chat_buffer) > self.chat_buffer_size:
            self.chat_buffer.pop(0)

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
            with Spinner(f"{active_agent.ai_name} is thinking... "):
                while len(active_agent.auditory_buffer) > 0:
                    agent_name, message = active_agent.auditory_buffer.pop(0)
                    active_agent.full_message_history.append(dict(
                        content=f"{agent_name}: {message}",
                        # Consider using the <name> field of the openai api instead of this format
                        role='user'
                    ))

                chat_with_ai_args = [active_agent.prompt,
                                     active_agent.user_input,
                                     active_agent.full_message_history,
                                     active_agent.memory,
                                     self.cfg.fast_token_limit]
                if USE_LMQL_QUERIES == True:
                    assistant_reply = lmql_utils.lmql_chat_with_ai(
                        *chat_with_ai_args)  # TODO: This hardcodes the model to use GPT3.5. Make this an argument
                else:
                    assistant_reply = chat_with_ai(
                        *chat_with_ai_args)  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            # Print Assistant thoughts

            assistant_reply_object = print_assistant_thoughts(active_agent.ai_name, assistant_reply)
            if assistant_reply_object is not None:
                try:
                    speak_value = assistant_reply_object.get('thoughts', {}).get('speak')
                    with Spinner(f"EVALUATING EMOTIONAL STATE OF {active_agent.ai_name}."):
                        val = lmql_utils.lmql_get_emotional_state(speak_value)
                    logger.typewriter_log(
                        "JARVIS: ", Fore.YELLOW,
                        f"Evaluation complete. {active_agent.ai_name} is feeling {val} right now."
                    )
                    successful = self.send_message_to_all_agents(speaker=active_agent, message=speak_value)
                    if successful:
                        # Only remove own message from buffer if it was non-empty
                        active_agent.auditory_buffer.pop()
                except Exception as e:
                    logger.error(f"Failed to add assistant reply to buffer.\n\n {e}\n\n")

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
                if not CHAT_ONLY_MODE:
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
                    # TODO add clean exit that closes event loop
                    # loop = asyncio.get_event_loop()
                    # loop.close()
                    break
            else:
                if not CHAT_ONLY_MODE:
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
                if not CHAT_ONLY_MODE:
                    logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            else:
                active_agent.full_message_history.append(
                    create_chat_message("system", "Unable to execute command")
                )
                if not CHAT_ONLY_MODE:
                    logger.typewriter_log(
                        "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
                    )
