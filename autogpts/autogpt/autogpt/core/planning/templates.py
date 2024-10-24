# Rules of thumb:
# - Templates don't add new lines at the end of the string.  This is the
#   responsibility of the or a consuming template.

####################
# Planner defaults #
####################


USER_OBJECTIVE = (
    "Write a wikipedia style article about the project: "
    "https://github.com/significant-gravitas/AutoGPT"
)


# Plan Prompt
# -----------


PLAN_PROMPT_CONSTRAINTS = (
    "~4000 word limit for short term memory. Your short term memory is short, so "
    "immediately save important information to files.",
    "If you are unsure how you previously did something or want to recall past "
    "events, thinking about similar events will help you remember.",
    "No user assistance",
    "Exclusively use the commands listed below e.g. command_name",
)

PLAN_PROMPT_RESOURCES = (
    "Internet access for searches and information gathering.",
    "Long-term memory management.",
    "File output.",
)

PLAN_PROMPT_PERFORMANCE_EVALUATIONS = (
    "Continuously review and analyze your actions to ensure you are performing to"
    " the best of your abilities.",
    "Constructively self-criticize your big-picture behavior constantly.",
    "Reflect on past decisions and strategies to refine your approach.",
    "Every command has a cost, so be smart and efficient. Aim to complete tasks in"
    " the least number of steps.",
    "Write all code to a file",
)


PLAN_PROMPT_RESPONSE_DICT = {
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user",
    },
    "command": {"name": "command name", "args": {"arg name": "value"}},
}

PLAN_PROMPT_RESPONSE_FORMAT = (
    "You should only respond in JSON format as described below\n"
    "Response Format:\n"
    "{response_json_structure}\n"
    "Ensure the response can be parsed by Python json.loads"
)

PLAN_TRIGGERING_PROMPT = (
    "Determine which next command to use, and respond using the format specified above:"
)

PLAN_PROMPT_MAIN = (
    "{header}\n\n"
    "GOALS:\n\n{goals}\n\n"
    "Info:\n{info}\n\n"
    "Constraints:\n{constraints}\n\n"
    "Commands:\n{commands}\n\n"
    "Resources:\n{resources}\n\n"
    "Performance Evaluations:\n{performance_evaluations}\n\n"
    "You should only respond in JSON format as described below\n"
    "Response Format:\n{response_json_structure}\n"
    "Ensure the response can be parsed by Python json.loads"
)


###########################
# Parameterized templates #
###########################
