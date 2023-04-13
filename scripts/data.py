from pathlib import Path


def load_prompt():
    """Load the prompt from data/prompt.txt"""
    try:
        # get directory of this file:
        file_dir = Path(__file__).parent
        prompt_file_path = file_dir / "data" / "prompt.txt"

        # Load the prompt from data/prompt.txt
        with open(prompt_file_path, "r") as prompt_file:
            prompt = prompt_file.read()

        return prompt
    except FileNotFoundError:
        print("Error: Prompt file not found", flush=True)
        return ""
