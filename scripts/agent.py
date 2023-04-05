import argparse
import json
import random
import sys
import time
import traceback
from enum import Enum, auto

import chat
import commands as cmd
import memory as mem
import openai
import token_counter
from ai_config import AIConfig
from colorama import Fore, Style
from config import Config
from helper import (load_variables, parse_arguments, print_assistant_thoughts,
                    print_to_console, prompt_user)
from json_parser import fix_and_parse_json
from llm_utils import create_chat_completion
from reflexion import construct_reflexion
from spinner import Spinner


def construct_prompt():
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

        # Get rid of this global:
        global ai_name
        ai_name = config.ai_name
        
        full_prompt = config.construct_full_prompt()
        return full_prompt

class Agent():
    def __init__(self):
        self.cfg = Config()
        parse_arguments()
        self.ai_name = ""
        self.initial_prompt = construct_prompt()
        self.full_message_history = []
        self.reflexions = []
        self.result = None
        self.user_input = "Determine which next command to use, and respond using the format specified above:"
        self.step_counter = 0
        self.max_steps = 100
        self.reflexion_frequency = 10


    def step(self):        
        current_context, tokens_remaining = self.build_agent_prompt(self.user_input)
        assistant_reply = self.prompt_agent(current_context, tokens_remaining)
            
        # print("assistant reply: "+assistant_reply)
        # Print Assistant thoughts
        print_assistant_thoughts(assistant_reply, self.ai_name)

        # Get command name and arguments
        try:
            command_name, arguments = cmd.get_command(assistant_reply)
        except Exception as e:
            print_to_console("Error: \n", Fore.RED, str(e))

        if not self.cfg.continuous_mode:
            ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
            # Get key press: Prompt the user to press enter to continue or escape
            # to exit
            self.user_input = ""
            print_to_console(
                "NEXT ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")
            print(
                f"Enter 'y' to authorise command or 'n' to exit program, or enter feedback for {ai_name}...",
                flush=True)
            while True:
                console_input = input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)
                if console_input.lower() == "y":
                    self.user_input = "GENERATE NEXT COMMAND JSON"
                    break
                elif console_input.lower() == "n":
                    self.user_input = "EXIT"
                    break
                else:
                    self.user_input = console_input
                    command_name = "human_feedback"
                    break

            if self.user_input == "GENERATE NEXT COMMAND JSON":
                print_to_console(
                "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                Fore.MAGENTA,
                "")
            elif self.user_input == "EXIT":
                print("Exiting...", flush=True)
        else:
            # Print command
            print_to_console(
                "NEXT ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")

        # Execute command
        if command_name.lower() == "error":
            result = f"Command {command_name} threw the following error: " + arguments
        elif command_name == "human_feedback":
            result = f"Human feedback: {self.user_input}"
        else:
            result = f"Command {command_name} returned: {cmd.execute_command(command_name, arguments)}"

        # Check if there's a result from the command append it to the message
        # history
        if result is not None:
            self.full_message_history.append(chat.create_chat_message("system", result))
            print_to_console("SYSTEM: ", Fore.YELLOW, result)
        else:
            self.full_message_history.append(
                chat.create_chat_message(
                    "system", "Unable to execute command"))
            print_to_console("SYSTEM: ", Fore.YELLOW, "Unable to execute command")

    

    def build_agent_prompt(self, user_input, debug=True):
        """
            Construct the prompt that will be sent to the Agent.
        """ 
        context_token_limit = self.cfg.fast_token_limit - 1000 # reserve 1000 tokens for response
        # load the initial prompt and the the
        # permantent memory into the current context
        current_context = [
                chat.create_chat_message("system", self.initial_prompt),
                chat.create_chat_message("system", f"Permanent memory: {mem.permanent_memory}")
        ]

        # Get message index
        self.next_message_index = len(self.full_message_history) - 1
        self.insertion_index = len(current_context)
        current_tokens_used = 0
        
        # Count the number of tokens in the current message
        current_tokens_used = token_counter.count_message_tokens(current_context, self.cfg.fast_llm_model)
        current_tokens_used += token_counter.count_message_tokens([chat.create_chat_message("user", user_input)], self.cfg.fast_llm_model)

        while self.next_message_index >= 0:
            message_to_add = self.full_message_history[self.next_message_index]
            tokens_to_add = token_counter.count_message_tokens([message_to_add], self.cfg.fast_llm_model)
            
            if current_tokens_used + tokens_to_add > context_token_limit:
                break

            current_context.insert(self.insertion_index, self.full_message_history[self.next_message_index])
            current_tokens_used += tokens_to_add

            self.next_message_index -= 1
        

        current_context.extend([chat.create_chat_message("user", user_input)]) # Append the user input to the context
        tokens_remaining = self.cfg.fast_token_limit - current_tokens_used # Calculate the number of tokens remaining

        if debug:
            print(f"Token limit: {self.cfg.fast_token_limit}")
            print(f"Send Token Count: {current_tokens_used}")
            print(f"Tokens remaining for response: {tokens_remaining}")
            print("------------ CONTEXT SENT TO AI ---------------")
            for message in current_context:
                # Skip printing the prompt
                # if message["role"] == "system" and message["content"] == prompt:
                #     continue
                print(
                    f"{message['role'].capitalize()}: {message['content']}")
                print()
            print("----------- END OF CONTEXT ----------------")

        return current_context, tokens_remaining
    
    def build_reflection_prompt(self, debug=True):
        reflect_context = [
            chat.create_chat_message("system", construct_reflexion()),
        ] # Create an empty reflection context

        current_tokens_spent = token_counter.count_message_tokens(reflect_context, self.cfg.reflect_llm_model)
        insertion_index = len(reflect_context)
        add_message_index = len(self.full_message_history) - 1

        # Add the previous reasoning trail to the reflection context 
        while add_message_index >= 0:

            message_to_add = self.full_message_history[add_message_index]
            tokens_to_add = token_counter.count_message_tokens([message_to_add], self.cfg.reflexion_llm_model)
            
            if current_tokens_spent + tokens_to_add > self.cfg.reflexion_llm_token_limit:
                break

            reflect_context.insert(insertion_index, self.full_message_history[add_message_index])
            current_tokens_spent += tokens_to_add

            add_message_index -= 1

        # Create reflection instruction
        reflect_context.extend(
            [chat.create_chat_message("user", "Your reflection for the reasoning trial above. (keep it short, 3 sentences max) Output format is: Reflection: <your reflection>")]
        )

        if debug:
            print("------------ REFLEXION CONTEXT---------------")
            for message in reflect_context:
                print(
                    f"{message['role'].capitalize()}: {message['content']}")
                print()
            print("----------- END OF REFLEXION ----------------")

        return reflect_context

    def prompt_agent(self, current_context, tokens_remaining):
        while True:
            try:
                reply = create_chat_completion(
                    model=self.cfg.fast_llm_model,
                    messages=current_context,
                    max_tokens=tokens_remaining,
                )
                return reply
            except openai.error.RateLimitError:
                # TODO: When we switch to langchain, this is built in
                print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
                time.sleep(10)


    def prompt_reflexion(self, reflexion_context):
        while True:
            try:
                reflection = create_chat_completion(
                    messages=reflexion_context,
                    model=self.cfg.reflect_llm_model,
                    max_tokens=1000
                )
                print("\n----------------- Reflection -----------------\n")
                print(reflection)
                print("\n-----------------------------------------------\n")
                self.reflexions.append(reflection)
                break;
            except openai.error.RateLimitError:
                    # TODO: When we switch to langchain, this is built in
                    print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
                    time.sleep(10)
       

    def reflect(self):
        reflection_prompt = self.build_reflection_prompt()
        self.prompt_reflexion(reflection_prompt)


    def run(self):
        while self.step_counter < self.max_steps:
            self.current_context = [] # Reset the current context each step.
            self.step()
            self.step_counter += 1

            # Reflect on trajectory every 5 steps
            if self.step_counter % self.reflexion_frequency == 0:
                self.reflect()


# test agent
if __name__ == "__main__":
    agent = Agent()
    agent.run()
