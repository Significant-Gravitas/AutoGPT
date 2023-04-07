import os
import uuid
from colorama import Fore, Style
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from app_types import StartBody, ChatBody
from app_utils import create_assistant_thoughts, create_message
from memory import PineconeMemory
from ai_config import AIConfig
from chat import chat_with_ai
from config import Config
from commands import get_command, execute_command
from chat import create_chat_message


app = FastAPI()
chat_data = {}
memory = PineconeMemory()
memory.clear()
print('Using memory of type: ' + memory.__class__.__name__)
cfg = Config()


def get_chat_workspace(chat_id: str) -> str:
    return os.path.join(".", "auto_gpt_workspace", chat_id)


@app.post("/start")
async def start_chat(request: Request, body: StartBody):
    chat_id = str(uuid.uuid4())
    chat_workspace = get_chat_workspace(chat_id)
    os.makedirs(chat_workspace, exist_ok=True)

    # set up chat context
    config = {
        "ai_name": body.ai_name,
        "ai_role": body.ai_role,
        "ai_goals": body.ai_goals,
    }
    prompt = AIConfig(**config).construct_full_prompt()
    user_input = "Determine which next command to use, and respond using the format specified above:"
    full_message_history = []

    messages = []

    # send message to AI, get response
    assistant_reply = chat_with_ai(
        prompt,
        user_input,
        full_message_history,
        memory,
        cfg.fast_token_limit,
        chat_id=chat_id,
    )

    # get assistant thoughts
    messages += create_assistant_thoughts(config.ai_name, assistant_reply)

    # get command name and arguments
    try:
        command_name, arguments = get_command(assistant_reply)
    except Exception as e:
        messages += create_message("Error: \n", Fore.RED, str(e))

    # needs authorization from user
    messages += create_message(
        "NEXT ACTION: ",
        Fore.CYAN,
        f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
    )

    # save chat context
    chat_data[chat_id] = {
        "config": config,
        "prompt": prompt,
        "full_message_history": full_message_history,
        "command_name": command_name,
        "arguments": arguments,
        "assistant_reply": assistant_reply,
    }

    response = Response()
    response.headers["chat_id"] = chat_id

    return {"messages": messages}


@app.post("/chat")
async def continue_chat(request: Request, body: ChatBody):
    chat_id = request.headers.get("chat_id")
    if not chat_id or chat_id not in chat_data:
        raise HTTPException(status_code=400, detail="Invalid chat_id")

    # get user message
    user_message = body.message

    # get chat context
    config = chat_data[chat_id]["config"]
    prompt = chat_data[chat_id]["prompt"]
    full_message_history = chat_data[chat_id]["full_message_history"]
    command_name = chat_data[chat_id]["command_name"]
    arguments = chat_data[chat_id]["arguments"]
    assistant_reply = chat_data[chat_id]["assistant_reply"]

    messages = []

    # parse user message
    if user_message.lower() == "y":
        user_input = "GENERATE NEXT COMMAND JSON"
    elif user_message.lower() == "n":
        user_input = "EXIT"
    else:
        user_input = user_message
        command_name = "human_feedback"

    if user_input == "GENERATE NEXT COMMAND JSON":
        messages += create_message(
            "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
            Fore.MAGENTA,
            ""
        )
    elif user_input == "EXIT":
        messages += create_message(
            "Shutting down chat...",
            Fore.MAGENTA,
            ""
        )
        chat_data.pop(chat_id)
        return {"messages": messages}

    # execute command
    if command_name == "task_complete":
        messages += create_message(
            "Shutting down chat...",
            Fore.MAGENTA,
            ""
        )
        chat_data.pop(chat_id)
        return {"messages": messages}

    if command_name.lower() == "error":
        result = f"Command {command_name} threw the following error: " + arguments
    elif command_name == "human_feedback":
        result = f"Human feedback: {user_input}"
    else:
        result = f"Command {command_name} returned: {execute_command(command_name, arguments, chat_id=chat_id)}"

    memory_to_add = f"Assistant Reply: {assistant_reply} " \
                    f"\nResult: {result} " \
                    f"\nHuman Feedback: {user_input} "

    memory.add(memory_to_add, chat_id=chat_id)

    # update message history
    if result is not None:
        full_message_history.append(create_chat_message("system", result))
        messages += create_message(
            "SYSTEM: ", Fore.YELLOW, result
        )
    else:
        full_message_history.append(
            create_chat_message(
                "system", "Unable to execute command"
            )
        )
        messages += create_message(
            "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
        )

    # send message to AI, get response
    assistant_reply = chat_with_ai(
        prompt,
        user_input,
        full_message_history,
        memory,
        cfg.fast_token_limit,
        chat_id=chat_id,
    )

    # get assistant thoughts
    messages += create_assistant_thoughts(config.ai_name, assistant_reply)

    # get command name and arguments
    try:
        command_name, arguments = get_command(assistant_reply)
    except Exception as e:
        messages += create_message("Error: \n", Fore.RED, str(e))

    # needs authorization from user
    messages += create_message(
        "NEXT ACTION: ",
        Fore.CYAN,
        f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}",
    )

    # save chat context
    chat_data[chat_id] = {
        "config": config,
        "prompt": prompt,
        "full_message_history": full_message_history,
        "command_name": command_name,
        "arguments": arguments,
        "assistant_reply": assistant_reply,
    }

    response = Response()
    response.headers["chat_id"] = chat_id

    return {"messages": messages}
