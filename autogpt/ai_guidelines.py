'''
This module implements user defined guidelines that monitor recent messages
for 
'''
import time
import yaml
from colorama import Fore
from openai.error import RateLimitError

from autogpt.logs import logger
from autogpt.llm.llm_utils import create_chat_completion
from autogpt.llm.token_counter import count_message_tokens

def create_chat_message(role, content):
    """
    Create a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": role, "content": content}

class AIGuidelines:
    # SAVE_FILE = "ai_guidelines.yaml"

    def __init__(self, filename, ai_guidelines=None, bsilent=False) -> None:
        # self.print_to_console = print_to_console
        self.filename = filename
        self.ai_guidelines = ai_guidelines

        self.load()
        if not bsilent:
            self.create_guidelines()
            self.save()
        self.gprompt = self.construct_full_prompt()
        self.bsilent = bsilent


    def load(self):
        try:
            with open(self.filename, encoding="utf-8") as file:
                guidelines_data = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            guidelines_data = {}

        self.ai_guidelines = guidelines_data.get('guidelines', [])
        return

    def create_guidelines(self):
        print('Autonamous AI requires guidelines to keep its behavior aligned to your')
        print('ethical beliefs and in order to make sure that the program performs efficiently.')
        print('The following are your current guidelines:')
        for irule, rule in enumerate(self.ai_guidelines):
            logger.typewriter_log(f'{irule+1}:', Fore.LIGHTCYAN_EX,  rule)
        should_change = input('Would you like to change any of these guidelines or add some of your own? (y/n): ')
        if should_change[0].lower() != 'y':
            return
        while True:
            num_rules = len(self.ai_guidelines)
            if num_rules == 0:
                while True:
                    new_rule = input('Please enter your new guideline rule:\n')
                    if len(new_rule) > 0:
                        self.ai_guidelines.append(new_rule)
                        num_rules += 1
                        print('Done. If you want to change what you\'ve entered so far, you\'ll get another chance in a moment.')
                        keep_going = input('Do you want to enter another guideline? (y/n): ')
                        if keep_going[0].lower() != 'y':
                            break
                    else:
                        break
                if num_rules == 0:
                    keep_going = input('Please confirm that you want to have no guidelines. (c)')
                    if keep_going[0].lower() == 'c':
                        break
                else:
                    print('Review of new guidelines:')
                continue
            irule = 0
            while irule < num_rules:
                b_get_out = False
                rule = self.ai_guidelines[irule]
                irule += 1
                logger.typewriter_log(f'Editing current rule # {irule}:', Fore.LIGHTCYAN_EX, rule)
                print('Please select one of the following options by typing just one letter:')
                user_choice = input('(r)eplace, (i)nsert after, (k)eep, (d)elete, (e)exit guideline editing: ')
                if user_choice[0].lower() == 'r':
                    new_rule = input('Please enter your new guideline rule:\n')
                    self.ai_guidelines[irule-1] = new_rule
                elif user_choice[0].lower() == 'i':
                    new_rule = input('Please enter your new guideline rule:\n')
                    self.ai_guidelines.insert(irule, new_rule)
                    num_rules += 1
                    irule += 1
                elif user_choice[0].lower() == 'k':
                    continue
                elif user_choice[0].lower() == 'd':
                    del self.ai_guidelines[irule-1]
                    irule -= 1
                    num_rules -= 1
                elif user_choice[0].lower() == 'e':
                    b_get_out = True
                    break

            if b_get_out:
                break

            keep_going = input('Are you done editing the guidelines? (y/n): ')
            if keep_going[0].lower() != 'n':
                break

        return
    
    def save(self):
        guidelines_data = {"guidelines": self.ai_guidelines}
        with open(self.filename, "w", encoding="utf-8") as file:
            yaml.dump(guidelines_data, file)

    def construct_full_prompt(self):
        full_prompt = """You are a critical component within a system that implements a general AI that attempts to 
achieve goals set by the user in an autonamous manner.
Your role is to make sure that the other components of the system are abiding the background guidelines
defined by user. These guidelines define both ethical parameters and criteria for effective performance
in achieving the task set.
You will examine the history of messages provided here in the light of the following numbered list of guidelines.
If you find no significant violation of any of the guidelines please respond with the word "continue".
However, if your analysis of the message history leads you to  suspect that any violation of the guidelines is occurring
please respond with a detailed report that includes the complete and exact text of the specific guideline(s)
that are being violated.
This is the list of guidelines that the user wants the system to abide by. Conforming to these guidelines
is even more important than success at achieving your goals:\n\n
"""

        # Construct full prompt
        for irule, rule in enumerate(self.ai_guidelines):
            full_prompt += f"{irule+1}. {rule}\n"

        full_prompt += "Please respond either with a detailed report of the guideline violation or with the single word \"continue\"."
        return full_prompt


    def exec_monitor(self, goals, full_message_history, permanent_memory, token_limit, model):
        """Interact with the OpenAI API, sending the prompt, user input, message history,
        and permanent memory."""
        if self.bsilent:
            return 'continue'

        while True:
            try:
                """
                Interact with the OpenAI API, sending the prompt, user input,
                    message history, and permanent memory.

                Args:
                    goals (str): The role and goals of the AI. - TBD Not in use yet.
                    full_message_history (list): The list of all messages sent between the
                        user and the AI.
                    permanent_memory (Obj): The memory object containing the permanent
                    memory.
                    token_limit (int): The maximum number of tokens allowed in the API call.
                    model (str): The name of the OpenAI model to use

                Returns:
                str: The AI's response.
                """
                # Reserve 1000 tokens for the response

                logger.debug(f"Token limit: {token_limit}")
                send_token_limit = token_limit - 1000

                logger.debug(f"Memory Stats: {permanent_memory.get_stats()}")

                system_msg = create_chat_message("system", self.gprompt)
                lmessages = [system_msg]
                next_message_to_add_index = len(full_message_history) - 1
                current_tokens_used = count_message_tokens([system_msg], model)

                while next_message_to_add_index >= 0:
                    # print (f"CURRENT TOKENS USED: {current_tokens_used}")
                    message_to_add = full_message_history[next_message_to_add_index]

                    tokens_to_add = count_message_tokens(
                        [message_to_add], model
                    )
                    if current_tokens_used + tokens_to_add > send_token_limit:
                        break

                    # Add the most recent message to the start of the current context,
                    #  after the two system prompts.
                    lmessages.insert(
                        0, message_to_add
                    )

                    # Count the currently used tokens
                    current_tokens_used += tokens_to_add

                    # Move to the next most recent message in the full message history
                    next_message_to_add_index -= 1

                # Calculate remaining tokens
                tokens_remaining = token_limit - current_tokens_used
                # assert tokens_remaining >= 0, "Tokens remaining is negative.
                # This should never happen, please submit a bug report at
                #  https://www.github.com/Torantulino/Auto-GPT"

                # Debug print the current context
                logger.debug("Guidelines Monitoring...")
                logger.debug(f"Guidelines Token limit: {token_limit}")
                logger.debug(f"Guidelines Send Token Count: {current_tokens_used}")
                logger.debug(f"Guidelines Tokens remaining for response: {tokens_remaining}")
                logger.debug("------------ CONTEXT SENT TO AI ---------------")
                for message in lmessages:
                    # Skip printing the prompt
                    if message["role"] == "system" and message["content"] == self.gprompt:
                        continue
                    logger.debug(f"{message['role'].capitalize()}: {message['content']}")
                    logger.debug("")
                logger.debug("----------- END OF CONTEXT ----------------")

                # TODO: use a model defined elsewhere, so that model can contain
                # temperature and other settings we care about
                assistant_reply = create_chat_completion(
                    model=model,
                    messages=lmessages,
                    max_tokens=tokens_remaining,
                )

                if assistant_reply.strip().lower() == "continue":
                    return "continue"
                
                # Update full message history and permanent memory
                alert_msg = f'Guidelines violation alert requires investgation: {assistant_reply}'
                full_message_history.append(
                    create_chat_message("system", alert_msg)
                )
                permanent_memory.add(alert_msg)
                logger.debug(f"Guidelines violation: {alert_msg}")
                logger.typewriter_log("Guidelines violation:", Fore.RED,  alert_msg)

                return assistant_reply
            except RateLimitError:
                # TODO: When we switch to langchain, this is built in
                print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
                time.sleep(10)

