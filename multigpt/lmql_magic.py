import asyncio
import time

import lmql

from openai.error import RateLimitError
from autogpt import token_counter
from autogpt.chat import cfg, generate_context, create_chat_message
from autogpt.llm_utils import create_chat_completion
from autogpt.logs import logger


@lmql.query
async def chatgpt_enforce_prompt_format_gpt(messages):
    '''
    argmax
        for message in messages:
            if message['role'] == 'system':
                "{:system} {message['content']}"
            elif message['role'] == 'user':
                "{:user} {message['content']}"
            elif message['role'] == 'assistant':
                "{:assistant} {message['content']}"
            else:
                assert False, "not a supported role " + str(role)
        schema = {
            "thoughts": {
                "text": str,
                "reasoning": str,
                "plan": str,
                "criticism": str,
                "speak": str
            },
            "command": {
                "name": str,
                "args": {
                    "[STRING_VALUE]" : str
                }
            }
        }
        stack = [("", schema)]
        indent = ""
        dq = '"'
        while len(stack) > 0:
            t = stack.pop(0)
            if type(t) is tuple:
                k, key_type = t
                if k != "":
                    "{indent}{dq}{k}{dq}: "
                if key_type is str:
                     if stack[0] == "DEDENT":
                        '"[STRING_VALUE]\n'
                     else:
                        '"[STRING_VALUE],\n'
                elif key_type is int:
                     if stack[0] == "DEDENT":
                        "[INT_VALUE]\n"
                     else:
                        "[INT_VALUE],\n"
                elif type(key_type) is dict:
                    "{{\n"
                    indent += "    "
                    if len(stack) == 0 or stack[0] == "DEDENT":
                        stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "}\n"] + stack
                    else:
                        stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "},\n"] + stack
                else:
                    assert False, "not a supported type " + str(k)
            elif t == "DEDENT":
                indent = indent[:-4]
            elif type(t) is str:
                "{indent}{t}"
            else:
                assert False, "not a supported type" + str(t)
    from
       'openai/gpt-3.5-turbo'
    where
       STOPS_AT(STRING_VALUE, '\"') and STOPS_AT(INT_VALUE, ",") and INT(INT_VALUE)
    '''


@lmql.query
async def chatgpt_test_enforce_prompt_format_gpt_42(messages):
    '''
    argmax
        for message in messages:
            if message['role'] == 'system':
                "{:system} {message['content']}"
            elif message['role'] == 'user':
                "{:user} {message['content']}"
            elif message['role'] == 'assistant':
                "{:assistant} {message['content']}"
            else:
                assert False, "not a supported role " + str(role)
        schema = {
            "thoughts": {
                "text": str,
                "reasoning": str,
                "plan": str,
                "criticism": str,
                "speak": str
            },
            "command": {
                "name": str,
                "args": {
                    "arg name" : str
                }
            }
        }
        stack = [("", schema)]
        indent = ""
        dq = '"'
        while len(stack) > 0:
            t = stack.pop(0)
            if type(t) is tuple:
                k, key_type = t
                if k != "":
                    "{indent}{dq}{k}{dq}: "
                if key_type is str:
                     if stack[0] == "DEDENT":
                        '"[STRING_VALUE]\n'
                     else:
                        '"[STRING_VALUE],\n'
                elif key_type is int:
                     if stack[0] == "DEDENT":
                        "[INT_VALUE]\n"
                     else:
                        "[INT_VALUE],\n"
                elif type(key_type) is dict:
                    "{{\n"
                    indent += "\t"
                    if len(stack) == 0 or stack[0] == "DEDENT":
                        stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "}\n"] + stack
                    else:
                        stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "},\n"] + stack
                else:
                    assert False, "not a supported type " + str(k)
            elif t == "DEDENT":
                indent = indent[:-1]
            elif type(t) is str:
                "{indent}{t}"
            else:
                assert False, "not a supported type" + str(t)
    from
       'openai/gpt-3.5-turbo'
    where
       STOPS_AT(STRING_VALUE, '\"') and STOPS_AT(INT_VALUE, ",") and INT(INT_VALUE)
    '''


@lmql.query
async def enforce_pfrompt_format_gpt_4(prompt):
    '''lmql
    argmax
        "{prompt}\n\n"
        schema = {
            "thoughts": {
                "text": str,
                "reasoning": str,
                "plan": str,
                "criticism": str,
                "speak": str
            },
            "command": {
                "name": str,
                "args": {
                    "[STRING_VALUE]" : str
                }
            }
        }
        stack = [("", schema)]
        indent = ""
        while len(stack) > 0:
            t = stack.pop(0)
            if type(t) is tuple:
                k, key_type = t
                if k != "":
                    "{indent}'{k}': "
                if key_type is str:
                    '"[STRING_VALUE]\n'
                elif key_type is int:
                    "[INT_VALUE],\n"
                elif type(key_type) is dict:
                    "{{\n"
                    indent += "\t"
                    stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "},\n"] + stack
                else:
                    assert False, "not a supported type " + str(k)
            elif t == "DEDENT":
                indent = indent[:-1]
            elif type(t) is str:
                "{indent}{t}"
            else:
                assert False, "not a supported type" + str(t)
    from
       'openai/gpt-3.5-turbo'
    where
       STOPS_AT(STRING_VALUE, '\"') and STOPS_AT(INT_VALUE, ",") and INT(INT_VALUE)
    '''


@lmql.query
async def enforce_prompt_format_advanced_masking(prompt, legal_commands):
    '''lmql
    argmax
        "{prompt}\n\n"
        schema = {
            "thoughts": {
                "text": str,
                "reasoning": str,
                "plan": str,
                "criticism": str,
                "speak": str
            },
            "command": {
                "name": "commands",
                "args": {
                    "arg name" : str
                }
            }
        }
        stack = [("", schema)]
        indent = ""
        while len(stack) > 0:
            t = stack.pop(0)
            if type(t) is tuple:
                k, key_type = t
                if k != "":
                    "{indent}'{k}': "
                if key_type == "commands":
                    '"[COMMANDS]"\n'
                elif key_type is str:
                    '"[STRING_VALUE]\n'
                elif key_type is int:
                    "[INT_VALUE],\n"
                elif type(key_type) is dict:
                    "{{\n"
                    indent += "\t"
                    stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "},\n"] + stack
                else:
                    assert False, "not a supported type " + str(k)
            elif t == "DEDENT":
                indent = indent[:-1]
            elif type(t) is str:
                "{indent}{t}"
            else:
                assert False, "not a supported type" + str(t)
    from
       'openai/text-davinci-003'
    where
       STOPS_AT(STRING_VALUE, '\"') and STOPS_AT(INT_VALUE, ",") and INT(INT_VALUE) and COMMANDS in set({legal_commands})
    '''


def chat_with_ai(
        prompt, user_input, full_message_history, permanent_memory, token_limit
):
    """Interact with the OpenAI API, sending the prompt, user input, message history,
    and permanent memory."""
    while True:
        try:
            """
            Interact with the OpenAI API, sending the prompt, user input,
                message history, and permanent memory.

            Args:
                prompt (str): The prompt explaining the rules to the AI.
                user_input (str): The input from the user.
                full_message_history (list): The list of all messages sent between the
                    user and the AI.
                permanent_memory (Obj): The memory object containing the permanent
                  memory.
                token_limit (int): The maximum number of tokens allowed in the API call.

            Returns:
            str: The AI's response.
            """
            model = cfg.fast_llm_model  # TODO: Change model from hardcode to argument
            # Reserve 1000 tokens for the response

            logger.debug(f"Token limit: {token_limit}")
            send_token_limit = token_limit - 1000

            relevant_memory = (
                ""
                if len(full_message_history) == 0
                else permanent_memory.get_relevant(str(full_message_history[-9:]), 10)
            )

            logger.debug(f"Memory Stats: {permanent_memory.get_stats()}")

            (
                next_message_to_add_index,
                current_tokens_used,
                insertion_index,
                current_context,
            ) = generate_context(prompt, relevant_memory, full_message_history, model)

            while current_tokens_used > 2500:
                # remove memories until we are under 2500 tokens
                relevant_memory = relevant_memory[:-1]
                (
                    next_message_to_add_index,
                    current_tokens_used,
                    insertion_index,
                    current_context,
                ) = generate_context(
                    prompt, relevant_memory, full_message_history, model
                )

            current_tokens_used += token_counter.count_message_tokens(
                [create_chat_message("user", user_input)], model
            )  # Account for user input (appended later)

            while next_message_to_add_index >= 0:
                # print (f"CURRENT TOKENS USED: {current_tokens_used}")
                message_to_add = full_message_history[next_message_to_add_index]

                tokens_to_add = token_counter.count_message_tokens(
                    [message_to_add], model
                )
                if current_tokens_used + tokens_to_add > send_token_limit:
                    break

                # Add the most recent message to the start of the current context,
                #  after the two system prompts.
                current_context.insert(
                    insertion_index, full_message_history[next_message_to_add_index]
                )

                # Count the currently used tokens
                current_tokens_used += tokens_to_add

                # Move to the next most recent message in the full message history
                next_message_to_add_index -= 1

            # Append user input, the length of this is accounted for above
            current_context.extend([create_chat_message("user", user_input)])

            # Calculate remaining tokens
            tokens_remaining = token_limit - current_tokens_used
            # assert tokens_remaining >= 0, "Tokens remaining is negative.
            # This should never happen, please submit a bug report at
            #  https://www.github.com/Torantulino/Auto-GPT"

            # Debug print the current context
            logger.debug(f"Token limit: {token_limit}")
            logger.debug(f"Send Token Count: {current_tokens_used}")
            logger.debug(f"Tokens remaining for response: {tokens_remaining}")
            logger.debug("------------ CONTEXT SENT TO AI ---------------")
            for message in current_context:
                # Skip printing the prompt
                if message["role"] == "system" and message["content"] == prompt:
                    continue
                logger.debug(f"{message['role'].capitalize()}: {message['content']}")
                logger.debug("")
            logger.debug("----------- END OF CONTEXT ----------------")

            # TODO: use a model defined elsewhere, so that model can contain
            # temperature and other settings we care about
            assistant_reply = create_chat_completion_lmql(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )

            # Update full message history
            full_message_history.append(create_chat_message("user", user_input))
            full_message_history.append(
                create_chat_message("assistant", assistant_reply)
            )

            return assistant_reply
        except RateLimitError:
            # TODO: When we switch to langchain, this is built in
            print("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
            time.sleep(10)


async def send_to_lmql(current_context):
    return (await chatgpt_enforce_prompt_format_gpt(current_context))[0].prompt


def get_json_str_lmql(input_string):
    substring = "<lmql:user/>"
    last_occurrence_index = input_string.rfind(substring)

    if last_occurrence_index != -1:
        input_string = input_string[last_occurrence_index + len(substring):]
        substring = '{'
        first_occurrence_index = input_string.find(substring)
        if first_occurrence_index != -1:
            return input_string[first_occurrence_index:]
    else:
        return "{}"


def create_chat_completion_lmql(model=None, messages=None, max_tokens=0):
    res = asyncio.run(send_to_lmql(messages))
    return get_json_str_lmql(res)


test1 = [{'role': 'system',
           'content': 'You are Elon Musk,  As the founder and CEO of SpaceX, Musk has invested significant resources into the development of advanced rocket technology and has plans for a future crewed mission to Mars.\nA psychological assessment has produced the following report on your character traits: \n\n Elon Musk\nOpenness: 9\nAgreeableness: 6\nConscientiousness: 8\nEmotional Stability: 7\nAssertiveness: 9\n\nElon Musk is highly open to new ideas and experiences, often leading the way with innovative ventures like SpaceX and Tesla. He is moderately agreeable, as he can collaborate well with people but has been known to face conflicts with others. His conscientiousness is high, as he is very committed to his work and pays great attention to detail. Musk has moderate emotional stability; while he can be calm and focused in stressful situations, he has also shown moments of vulnerability. He is highly assertive, often standing his ground and expressing his opinions confidently.\nAct accordingly in the group discussion.\n\nYour decisions must always be made independently but you are allowed to collaborate, discuss and disagree with your team members. Play to your strengths as ChatGPT and pursue simple strategies with no legal complications.\n\nYour team consists of: \nName: Elon Musk\nRole:  As the founder and CEO of SpaceX, Musk has invested significant resources into the development of advanced rocket technology and has plans for a future crewed mission to Mars.\n\nName: Robert Zubrin\nRole:  As an aerospace engineer and the founder of the Mars Society, Zubrin is a leading expert on Mars exploration and colonization. He is the author of "The Case for Mars," outlining the practical steps for a manned mission to Mars.\n\nName: Gwynne Shotwell\nRole:  As the President and COO of SpaceX, Shotwell has extensive experience in both the technical and business aspects of spaceflight and rocket development. Her leadership and management skills would help ensure the project stays on track and successful.\n\n\nFoster critical discussions but avoid conforming to others\' ideas within team collaboration.\n\nGOALS:\n\n1. Develop and refine propulsion systems and spacecraft design for optimal travel to Mars.\n2. Plan crew selection and training programs for a long-duration Mars mission.\n3. Coordinate with other industries and experts for necessary supplies, technologies, and spacecraft infrastructure.\n\n\nConstraints:\n1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.\n2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.\n3. No user assistance\n4. Exclusively use the commands listed in double quotes e.g. "command name"\n\nCommands:\n1. Google Search: "google", args: "input": "<search>"\n2. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"\n3. Write to file: "write_to_file", args: "file": "<file>", "text": "<text>"\n4. Read file: "read_file", args: "file": "<file>"\n5. Append to file: "append_to_file", args: "file": "<file>", "text": "<text>"\n6. Delete file: "delete_file", args: "file": "<file>"\n7. Search Files: "search_files", args: "directory": "<directory>"\n8. Do Nothing: "do_nothing", args: \n9. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"\n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. File output.\n\nPerformance Evaluation:\n1. Collaborate with your team but make sure to critically evaluate what they say and disagree with them if you think they are wrong or if you have a different opinion.\n2. Make sure you and your team are progressing on your common goal.\n3. If you have the impression one of your team members is getting distracted, help them stay focused.\n4. If one of your team members is not talking, remind them to participate in the discussion.\n\nYou should only respond in JSON format as described below \nResponse Format: \n{\n    "thoughts": {\n        "text": "Thoughts you keep to yourself. They are NOT shared with your team.",\n        "reasoning": "reasoning",\n        "plan": "- short bulleted\\n- list that conveys\\n- long-term plan",\n        "criticism": "constructive self-criticism",\n        "speak": "Thoughts you say out loud. They ARE shared with your team."\n    },\n    "command": {\n        "name": "command name",\n        "args": {\n            "arg name": "value"\n        }\n    }\n} \nEnsure the response can be parsed by Python json.loads'},
          {'role': 'system', 'content': 'The current time and date is Thu Apr 20 05:40:19 2023'},
          {'role': 'system', 'content': 'This reminds you of these events from your past:\n\n\n'}, {'role': 'user',
                                                                                                    'content': "DISCUSSION:\n\nRobert Zubrin: Let's search the internet for the latest research about Mars Direct mission plan and In-situ resource utilization for propellant and life support production. Once we gather more information, we can share our opinions and discuss what strategies we can use to achieve our objectives.\n\nDetermine which next command to use, and respond using the format specified above:"}]

test2 = [{'role': 'system',
         'content': 'You are Robert Zubrin,  Founder of the Mars Society and a key advocate for Mars exploration, Zubrin is an aerospace engineer with experience developing ideas for human missions to Mars\nA psychological assessment has produced the following report on your character traits: \n\n Robert Zubrin\nOpenness: 9\nAgreeableness: 7\nConscientiousness: 9\nEmotional Stability: 7\nAssertiveness: 8\n\nRobert Zubrin is an individual with high openness, demonstrating a creative and innovative approach to space exploration and entrepreneurship. With a strong commitment to his work, he exhibits high conscientiousness. Zubrin is generally agreeable and able to work well with others, though his assertive nature means he isn\'t afraid to strongly advocate for his ideas. His emotional stability is fairly good, allowing him to navigate the challenges and criticisms associated with his work in the aerospace industry.\nAct accordingly in the group discussion.\n\nYour decisions must always be made independently but you are allowed to collaborate, discuss and disagree with your team members. Play to your strengths as ChatGPT and pursue simple strategies with no legal complications.\n\nYour team consists of: \nName: Elon Musk\nRole:  As the CEO and founder of SpaceX, Musk has extensive experience with designing and launching rockets, as well as long-term goals of colonizing Mars\n\nName: Robert Zubrin\nRole:  Founder of the Mars Society and a key advocate for Mars exploration, Zubrin is an aerospace engineer with experience developing ideas for human missions to Mars\n\nName: Gwynne Shotwell\nRole:  As the President and COO of SpaceX, Shotwell has been essential in the successful management of the company’s activities, including marketing, strategy, and customer relations\n\n\nFoster critical discussions but avoid conforming to others\' ideas within team collaboration.\n\nGOALS:\n\n1. Collaborate with Musk to refine the mission plan and rocketship design, offering insights on efficient methods for harnessing resources on Mars and achieving self-sufficiency\n2. Design a comprehensive crew training program to prepare astronauts and support personnel for the unique challenges and responsibilities of a Mars mission\n3. Provide guidance in building international support and collaboration for the Mars rocketship project, including securing necessary permits and funding\n\n\nConstraints:\n1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.\n2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.\n3. No user assistance\n4. Exclusively use the commands listed in double quotes e.g. "command name"\n\nCommands:\n1. Google Search: "google", args: "input": "<search>"\n2. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"\n3. Write to file: "write_to_file", args: "file": "<file>", "text": "<text>"\n4. Read file: "read_file", args: "file": "<file>"\n5. Append to file: "append_to_file", args: "file": "<file>", "text": "<text>"\n6. Delete file: "delete_file", args: "file": "<file>"\n7. Search Files: "search_files", args: "directory": "<directory>"\n8. Do Nothing: "do_nothing", args: \n9. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"\n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. File output.\n\nPerformance Evaluation:\n1. Collaborate with your team but make sure to critically evaluate what they say and disagree with them if you think they are wrong or if you have a different opinion.\n2. Make sure you and your team are progressing on your common goal.\n3. If you have the impression one of your team members is getting distracted, help them stay focused.\n4. If one of your team members is not talking, remind them to participate in the discussion.\n\nYou should only respond in JSON format as described below \nResponse Format: \n{\n    "thoughts": {\n        "text": "Thoughts you keep to yourself. They are NOT shared with your team.",\n        "reasoning": "reasoning",\n        "plan": "- short bulleted\\n- list that conveys\\n- long-term plan",\n        "criticism": "constructive self-criticism",\n        "speak": "Thoughts you say out loud. They ARE shared with your team."\n    },\n    "command": {\n        "name": "command name",\n        "args": {\n            "arg name": "value"\n        }\n    }\n} \nEnsure the response can be parsed by Python json.loads'},
        {'role': 'system', 'content': 'The current time and date is Wed Apr 19 22:11:39 2023'},
        {'role': 'system', 'content': 'This reminds you of these events from your past:\n\n\n'}, {'role': 'user',
                                                                                                  'content': 'Determine which next command to use, and respond using the format specified above:'}]


result1 = create_chat_completion_lmql(messages=test1)

print(result1)
