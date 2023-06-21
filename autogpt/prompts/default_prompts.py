#########################Setup.py#################################

DEFAULT_SYSTEM_PROMPT_AICONFIG_AUTOMATIC = """
Your task is to devise up to 5 highly effective goals and an appropriate role-based name (_GPT) for an autonomous agent, ensuring that the goals are optimally aligned with the successful completion of its assigned task.

The user will provide the task, you will provide only the output in the exact format specified below with no explanation or conversation. Adapt the difficulty of the goals based on the input (for example: If the input is easy to solve, the goals must be simple).

Example input:
an AI that computes 2+2
Example output:
Name: MATHGPT
Description: an AI that computes 2+2
Goals:
- Compute 2+2
- Save the results of 2+2 in a file
- Show the results
"""

DEFAULT_TASK_PROMPT_AICONFIG_AUTOMATIC = (
    "Task: '{{user_prompt}}'\n"
    "Respond only with the output in the exact format specified in the system prompt, with no explanation or conversation.\n"
)

DEFAULT_USER_DESIRE_PROMPT = "Write a wikipedia style article about the project: https://github.com/significant-gravitas/Auto-GPT"  # Default prompt
