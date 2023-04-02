from pathlib import Path

FOLDER = Path(__file__).parent


def load_prompt():
    try:
        # Load the promt from data/prompt.txt
        with open(FOLDER / "data/prompt.txt", "r") as prompt_file:
            prompt = prompt_file.read()

        return prompt
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""
