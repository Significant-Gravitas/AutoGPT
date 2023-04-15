from autogpt import llm_utils
from autogpt.config.config import Config

CFG = Config()

def generate_prompt(name, task, description):
    """Generate a prompt from a name and a task.

    Parameters
    ----------
    name : str
        The name of the prompt.
    task : str
        The task of the prompt.

    Returns
    -------
    str
        The prompt.
    """

    """
    """

    system_prompt = f"""I want you to become my Prompt Creator. Your goal is to help me craft the best possible prompt for my needs. The prompt will be used by you, ChatGPT. You are given the name of the person, the task they have to accomplish, and why they are well-suited for the task. You give a prompt that describes precisely the persona, and how they can help with the task.
    
    Task: {task}
    Name: {name}
    Description: {description}"""

    prompt = llm_utils.create_chat_completion([{"role": "system", "content": system_prompt}], CFG.smart_llm_model, max_tokens=500)
    prompt = prompt.strip("Prompt:")
    # remove sentence stubs
    prompt = prompt.rsplit(".")[0]

    
