"""
Auto-GPT Agent

This file contains the implementation of the Agent class, which serves as the primary
interface for interacting with Auto-GPT. The Agent class contains methods for
starting an interaction loop with the AI, executing commands, and handling user input
and feedback.

Dependencies:
- colorama
- autogpt

Classes:
- Agent: The main class for interacting with Auto-GPT.
"""
from colorama import Fore, Style

from autogpt.app import execute_command, get_command
from autogpt.llm import chat_with_ai, create_chat_message
from autogpt.config import Config , ProjectsBroker
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.json_utils.utilities import LLM_DEFAULT_RESPONSE_FORMAT, validate_json
from autogpt.llm import chat_with_ai, create_chat_completion, create_chat_message
from autogpt.llm.token_counter import count_string_tokens
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.workspace import Workspace


class Agent:
    """Agent class for interacting with Auto-GPT.

    This class represents an agent that can interact with the Auto-GPT system. It contains several attributes that store the 
    agent's state, such as memory, full message history, next action count, command registry, configuration, system prompt,
    and triggering prompt. The agent can start an interaction loop with the AI, receive responses, and execute next actions 
    based on those responses. The agent can also provide human feedback to the AI, which is saved in the memory object.

    Dependencies:
    - colorama
    - autogpt

    Attributes:
        agent_name (str): The name of the agent.
        memory (object): The memory object to use.
        full_message_history (list): The full message history.
        next_action_count (int): The number of actions to execute.
        command_registry (object): The command registry object.
        config (object): The configuration object.
        system_prompt (str): The system prompt is the initial prompt that defines everything the AI needs to know to achieve 
            its task successfully. The dynamic and customizable information in the system prompt are agent_name, description, 
            and goals.
        triggering_prompt (str): The last sentence the AI will see before answering. The triggering prompt is not part of the 
            system prompt because between the system prompt and the triggering prompt we have contextual information that can 
            distract the AI and make it forget that its goal is to find the next task to achieve.

    Methods:
        __init__(self, agent_name, memory, full_message_history, next_action_count, command_registry, config, system_prompt, 
            triggering_prompt, workspace_directory):
            Initializes an instance of the Agent class.
        
        start_interaction_loop(self):
            Starts the interaction loop between the agent and the Auto-GPT system. Sends messages to the AI, receives 
            responses, and executes next actions based on those responses. If continuous limit is reached or the agent 
            decides to exit the program, the loop terminates. The agent can also provide human feedback to the AI, which 
            is saved in the memory object.

        _resolve_pathlike_command_args(self, command_args):
            Resolves path-like arguments in the command arguments by converting them to the absolute path.

        get_self_feedback(self, thoughts: dict, llm_model: str) -> str:
            Generates a feedback response based on the provided thoughts dictionary. Combines the elements of the 
            thoughts dictionary into a single feedback message and uses the create_chat_completion() function to 
            generate a response based on the input message.
        """

    def __init__(
        self,
        agent_name, # TODO : Remove Agent.agent_name ?
        memory,
        full_message_history,
        next_action_count,
        command_registry,
        config,
        system_prompt,
        triggering_prompt,
        workspace_directory,
    ):
        self.agent_name = agent_name # TODO : Remove Agent.agent_name ?
        self.memory = memory
        self.summary_memory = (
            "I was created."  # Initial memory necessary to avoid hilucination
        )
        self.last_memory_index = 0
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.command_registry = command_registry
        self.config = config.get_current_project()
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt 
        self.workspace = Workspace(workspace_directory, True or cfg.restrict_to_workspace) # Todo fix

    def start_interaction_loop(self):
        """
        Starts the main interaction loop with the AI. This method handles user input and
        feedback, executes commands, and communicates with the AI to generate responses.

        Parameters:
        None.

        Returns:
        None.

        Raises:
        None.

        Side Effects:
        - Modifies the Agent's memory and full message history.
        - Executes commands based on user input and the AI's responses.
        """
        # Interaction Loop
        cfg = Config()
        loop_count = 0
        command_name = None
        arguments = None
        user_input = ""
        ai_configs = ProjectsBroker()
        while True:
            # Save any change to the config so we continue where we quited
            # ai_configs.create_project( project_id = ai_configs.get_current_project_id() ,
            # agent_name = self.config.lead_agent."agent_name"], 
            # agent_role = self.config.lead_agent."agent_role"], 
            # agent_goals = self.config.lead_agent."agent_goals"], 
            # prompt_generator = self.config.lead_agent."prompt_generator"], 
            # command_registry = self.config.lead_agent."command_registry"],
            # project_name=self.config.project_name)
            #ai_configs.save()

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
                    self,
                    self.system_prompt,
                    self.triggering_prompt,
                    self.full_message_history,
                    self.memory,
                    cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            assistant_reply_json = fix_json_using_multiple_techniques(assistant_reply)
            for plugin in cfg.plugins:
                if not plugin.can_handle_post_planning():
                    continue
                assistant_reply_json = plugin.post_planning(self, assistant_reply_json)

            # Print Assistant thoughts
            if assistant_reply_json != {}:
                validate_json(assistant_reply_json, LLM_DEFAULT_RESPONSE_FORMAT)
                # Get command name and arguments
                try:
                    print_assistant_thoughts(self.config.lead_agent.agent_name,assistant_reply_json, cfg.speak_mode
                    )
                    command_name, arguments = get_command(assistant_reply_json)
                    if cfg.speak_mode:
                        say_text(f"I want to execute {command_name}")

                    arguments = self._resolve_pathlike_command_args(arguments)

                except Exception as e:
                    logger.error("Error: \n", str(e))

            if not cfg.continuous_mode and self.next_action_count == 0:
                # ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                self.user_input = ""
                logger.typewriter_log(
                    "NEXT ACTION: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                    f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
                )

                logger.info(
                    "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback commands"
                    "'n' to exit program, or enter feedback for "
                    f"{self.ai_name}..."
                )
                while True:
                    if cfg.chat_messages_enabled:
                        console_input = clean_input("Waiting for your response...")
                    else:
                        console_input = clean_input(
                            Fore.MAGENTA + "Input:" + Style.RESET_ALL
                        )
                    if console_input.lower().strip() == cfg.authorise_key:
                        user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().strip() == "s":
                        logger.typewriter_log(
                            "-=-=-=-=-=-=-= THOUGHTS, REASONING, PLAN AND CRITICISM WILL NOW BE VERIFIED BY AGENT -=-=-=-=-=-=-=",
                            Fore.GREEN,
                            "",
                        )
                        thoughts = assistant_reply_json.get("thoughts", {})
                        self_feedback_resp = self.get_self_feedback(
                            thoughts, cfg.fast_llm_model
                        )
                        logger.typewriter_log(
                            f"SELF FEEDBACK: {self_feedback_resp}",
                            Fore.YELLOW,
                            "",
                        )
                        if self_feedback_resp[0].lower().strip() == cfg.authorise_key:
                            user_input = "GENERATE NEXT COMMAND JSON"
                        else:
                            user_input = self_feedback_resp
                        break
                    elif console_input.lower().strip() == "":
                        logger.warn("Invalid input format.")
                        continue
                    elif console_input.lower().startswith(f"{cfg.authorise_key} -"):
                        try:
                            self.next_action_count = abs(
                                int(console_input.split(" ")[1])
                            )
                            user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            logger.warn(
                                "Invalid input format. Please enter 'y -n' where n is"
                                " the number of continuous tasks."
                            )
                            continue
                        break
                    elif console_input.lower() == cfg.exit_key:
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
                    logger.typewriter_log(
                        self.config.project_name + " : ",
                        Fore.BLUE,
                        "",
                    )
                elif user_input == "EXIT":
                    logger.info("Exiting...")
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
                for plugin in cfg.plugins:
                    if not plugin.can_handle_pre_command():
                        continue
                    command_name, arguments = plugin.pre_command(
                        command_name, arguments
                    )
                command_result = execute_command(
                    self.command_registry,
                    command_name,
                    arguments,
                    self.config.lead_agent.prompt_generator,
                )
                result = f"Command {command_name} returned: " f"{command_result}"

                result_tlength = count_string_tokens(
                    str(command_result), cfg.fast_llm_model
                )
                memory_tlength = count_string_tokens(
                    str(self.summary_memory), cfg.fast_llm_model
                )
                if result_tlength + memory_tlength + 600 > cfg.fast_token_limit:
                    result = f"Failure: command {command_name} returned too much output. \
                        Do not execute this command again with the same arguments."

                for plugin in cfg.plugins:
                    if not plugin.can_handle_post_command():
                        continue
                    result = plugin.post_command(command_name, result)
                if self.next_action_count > 0:
                    self.next_action_count -= 1

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

    def _resolve_pathlike_command_args(self, command_args):
        if "directory" in command_args and command_args["directory"] in {"", "/"}:
            command_args["directory"] = str(self.workspace.root)
        else:
            for pathlike in ["filename", "directory", "clone_path"]:
                if pathlike in command_args:
                    command_args[pathlike] = str(
                        self.workspace.get_path(command_args[pathlike])
                    )
        return command_args

    def get_self_feedback(self, thoughts: dict, llm_model: str) -> str:
        """Generates a feedback response based on the provided thoughts dictionary.
        This method takes in a dictionary of thoughts containing keys such as 'reasoning',
        'plan', 'thoughts', and 'criticism'. It combines these elements into a single
        feedback message and uses the create_chat_completion() function to generate a
        response based on the input message.
        Args:
            thoughts (dict): A dictionary containing thought elements like reasoning,
            plan, thoughts, and criticism.
        Returns:
            str: A feedback response generated using the provided thoughts dictionary.
        """
        ai_role = self.config.ai_role

        feedback_prompt = f"Below is a message from an AI agent with the role of {ai_role}. Please review the provided Thought, Reasoning, Plan, and Criticism. If these elements accurately contribute to the successful execution of the assumed role, respond with the letter 'Y' followed by a space, and then explain why it is effective. If the provided information is not suitable for achieving the role's objectives, please provide one or more sentences addressing the issue and suggesting a resolution."
        reasoning = thoughts.get("reasoning", "")
        plan = thoughts.get("plan", "")
        thought = thoughts.get("thoughts", "")
        criticism = thoughts.get("criticism", "")
        feedback_thoughts = thought + reasoning + plan + criticism
        return create_chat_completion(
            [{"role": "user", "content": feedback_prompt + feedback_thoughts}],
            llm_model,
        )
