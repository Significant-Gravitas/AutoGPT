import json

def print_to_console(
        title,
        content="",
        speak_text=False,
        min_typing_speed=0.05,
        max_typing_speed=0.01):
    object = {
        "title": title,
        "content": content
    }
    print(json.dumps(object) + ",", flush=True)