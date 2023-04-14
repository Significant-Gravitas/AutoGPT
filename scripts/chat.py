import time
import openai
from dotenv import load_dotenv
from config import Config
import token_counter
from llm_utils import create_chat_completion
from logger import logger
import logging

cfg = Config()


def create_chat_message(role, content):
    """
    创建具有给定角色和内容的聊天消息。

     参数：
     角色 (str)：消息发送者的角色，例如“系统”、“用户”或“助手”。
     content (str): 消息的内容。

    Returns:
    dict: 包含消息的作用和内容的字典。
    """
    return {"role": role, "content": content}


def generate_context(prompt, relevant_memory, full_message_history, model):
    current_context = [
        create_chat_message(
            "system", prompt),
        create_chat_message(
            "system", f"现在的时间和日期是 {time.strftime('%c')}"),
        create_chat_message(
            "system", f"这让你想起了你过去的某些事件:\n{relevant_memory}\n\n")]

    # Add messages from the full message history until we reach the token limit
    next_message_to_add_index = len(full_message_history) - 1
    insertion_index = len(current_context)
    # Count the currently used tokens
    current_tokens_used = token_counter.count_message_tokens(current_context, model)
    return next_message_to_add_index, current_tokens_used, insertion_index, current_context


# TODO: Change debug from hardcode to argument
def chat_with_ai(
        prompt,
        user_input,
        full_message_history,
        permanent_memory,
        token_limit):
    """与 OpenAI API 交互，发送提示、用户输入、消息历史记录和永久内存。"""
    while True:
        try:
            """
            与 OpenAI API 交互，发送提示、用户输入、消息历史记录和永久内存。

             参数：
             prompt (str): 向 AI 解释规则的提示。
             user_input (str)：来自用户的输入。
             full_message_history (list)：用户和AI之间发送的所有消息的列表。
             permanent_memory (Obj)：包含永久内存的内存对象。
             token_limit (int)：API 调用中允许的最大令牌数。

            Returns:
            str: AI的回应.
            """
            model = cfg.fast_llm_model # TODO: Change model from hardcode to argument
            # Reserve 1000 tokens for the response

            logger.debug(f"Token 限制: {token_limit}")
            send_token_limit = token_limit - 1000

            relevant_memory = '' if len(full_message_history) ==0 else  permanent_memory.get_relevant(str(full_message_history[-9:]), 10)

            logger.debug(f'内存状态: {permanent_memory.get_stats()}')

            next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                prompt, relevant_memory, full_message_history, model)

            while current_tokens_used > 2500:
                # remove memories until we are under 2500 tokens
                relevant_memory = relevant_memory[1:]
                next_message_to_add_index, current_tokens_used, insertion_index, current_context = generate_context(
                    prompt, relevant_memory, full_message_history, model)

            current_tokens_used += token_counter.count_message_tokens([create_chat_message("user", user_input)], model) # Account for user input (appended later)

            while next_message_to_add_index >= 0:
                # print (f"CURRENT TOKENS USED: {current_tokens_used}")
                message_to_add = full_message_history[next_message_to_add_index]

                tokens_to_add = token_counter.count_message_tokens([message_to_add], model)
                if current_tokens_used + tokens_to_add > send_token_limit:
                    break

                # Add the most recent message to the start of the current context, after the two system prompts.
                current_context.insert(insertion_index, full_message_history[next_message_to_add_index])

                # Count the currently used tokens
                current_tokens_used += tokens_to_add

                # Move to the next most recent message in the full message history
                next_message_to_add_index -= 1

            # Append user input, the length of this is accounted for above
            current_context.extend([create_chat_message("user", user_input)])

            # Calculate remaining tokens
            tokens_remaining = token_limit - current_tokens_used
            # assert tokens_remaining >= 0, "Tokens remaining is negative. This should never happen, please submit a bug report at https://www.github.com/Torantulino/Auto-GPT"

            # Debug print the current context
            logger.debug(f"Token 限制: {token_limit}")
            logger.debug(f"发送Token 数量: {current_tokens_used}")
            logger.debug(f"Tokens 剩余回应: {tokens_remaining}")
            logger.debug("------------ 发送给 AI 的上下文信息 ---------------")
            for message in current_context:
                # Skip printing the prompt
                if message["role"] == "system" and message["content"] == prompt:
                    continue
                logger.debug(f"{message['role'].capitalize()}: {message['content']}")
                logger.debug("")
            logger.debug("----------- 结束上下文信息 ----------------")

            # TODO: use a model defined elsewhere, so that model can contain temperature and other settings we care about
            assistant_reply = create_chat_completion(
                model=model,
                messages=current_context,
                max_tokens=tokens_remaining,
            )

            # Update full message history
            full_message_history.append(
                create_chat_message(
                    "user", user_input))
            full_message_history.append(
                create_chat_message(
                    "assistant", assistant_reply))

            return assistant_reply
        except openai.error.RateLimitError:
            # TODO: When we switch to langchain, this is built in
            print("Error: ", "已达到 API 速率限制。 等待 10 秒...")
            time.sleep(10)
