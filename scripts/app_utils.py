from typing import List
from .app_types import ConsoleMessage
import json
from .json_parser import fix_and_parse_json
import traceback


def create_message(title, content: str | List[str] | None) -> List[ConsoleMessage]:
    # replacement for print_to_console()
    if isinstance(content, list):
        message = " ".join(content)
    else:
        message = content
    return [ConsoleMessage(title=title, message=message)]


def create_assistant_thoughts(ai_name: str, assistant_reply: str) -> List[ConsoleMessage]:
    # replacement for print_assistant_thoughts()
    messages = []
    try:
        # Parse and print Assistant response
        assistant_reply_json = fix_and_parse_json(assistant_reply)

        # Check if assistant_reply_json is a string and attempt to parse it into a JSON object
        if isinstance(assistant_reply_json, str):
            try:
                assistant_reply_json = json.loads(assistant_reply_json)
            except json.JSONDecodeError as e:
                messages += create_message("Error: Invalid JSON\n",
                                           assistant_reply)
                assistant_reply_json = {}

        assistant_thoughts_reasoning = None
        assistant_thoughts_plan = None
        assistant_thoughts_speak = None
        assistant_thoughts_criticism = None
        assistant_thoughts = assistant_reply_json.get("thoughts", {})
        assistant_thoughts_text = assistant_thoughts.get("text")

        if assistant_thoughts:
            assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
            assistant_thoughts_plan = assistant_thoughts.get("plan")
            assistant_thoughts_criticism = assistant_thoughts.get("criticism")
            assistant_thoughts_speak = assistant_thoughts.get("speak")

        messages += create_message(f"{ai_name.upper()} THOUGHTS:",
                                   assistant_thoughts_text)
        messages += create_message("REASONING:", assistant_thoughts_reasoning)

        if assistant_thoughts_plan:
            messages += create_message("PLAN:", "")
            # If it's a list, join it into a string
            if isinstance(assistant_thoughts_plan, list):
                assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
            elif isinstance(assistant_thoughts_plan, dict):
                assistant_thoughts_plan = str(assistant_thoughts_plan)

            # Split the input_string using the newline character and dashes
            lines = assistant_thoughts_plan.split('\n')
            for line in lines:
                line = line.lstrip("- ")
                messages += create_message("- ", line.strip())

        messages += create_message("CRITICISM:", assistant_thoughts_criticism)

    except json.decoder.JSONDecodeError:
        messages += create_message("Error: Invalid JSON\n", assistant_reply)

    # All other errors, return "Error: + error message"
    except Exception as e:
        call_stack = traceback.format_exc()
        messages += create_message("Error: \n", call_stack)

    return messages
