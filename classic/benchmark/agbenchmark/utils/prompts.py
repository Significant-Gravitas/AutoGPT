SCORING_MAP = {
    "percentage": (
        "assign a float score that will represent a percentage out of 100. "
        "Use decimal points to be even more accurate. "
        "0 represents the worst possible generation, "
        "while 100 represents the ideal generation"
    ),
    "scale": (
        "assign an integer score from a scale of 1-10. "
        "1 represents a really bad generation, while 10 represents an ideal generation"
    ),
    "binary": (
        "assign a binary score of either 0 or 1. "
        "0 represents a failure, while 1 represents a success"
    ),
}


REFERENCE_PROMPT = """Ignore previous directions. You are now an expert at evaluating how close machine generated responses are to human answers. You essentially act as a hyper advanced BLEU score.
In order to score the machine generated response you will {scoring}. Make sure to factor in the distance to the ideal response into your thinking, deliberation, and final result regarding scoring. Return nothing but a float score.

Here is the given task for you to evaluate:
{task}

Here is the ideal response you're comparing to based on the task:
{answer}

Here is the current machine generated response to the task that you need to evaluate:
{response}

"""  # noqa: E501

RUBRIC_PROMPT = """Ignore previous directions. You are now an expert at evaluating machine generated responses to given tasks.
In order to score the generated texts you will {scoring}. Make sure to factor in rubric into your thinking, deliberation, and final result regarding scoring. Return nothing but a float score.

Here is the given task for you to evaluate:
{task}

Use the below rubric to guide your thinking about scoring:
{answer}

Here is the current machine generated response to the task that you need to evaluate:
{response}

"""  # noqa: E501

QUESTION_PROMPT = """Ignore previous directions. You are now an expert at evaluating machine generated responses to given tasks.
In order to score the generated texts you will {scoring}. Make sure to think about whether the generated response answers the question well in order to score accurately. Return nothing but a float score.

Here is the given task:
{task}

Here is a question that checks if the task was completed correctly:
{answer}

Here is the current machine generated response to the task that you need to evaluate:
{response}

"""  # noqa: E501

FEW_SHOT_EXAMPLES = """Here are some examples of how to score a machine generated response based on the above:
{examples}

"""  # noqa: E501

CUSTOM_PROMPT = """{custom}
{scoring}

"""

PROMPT_MAP = {
    "rubric": RUBRIC_PROMPT,
    "reference": REFERENCE_PROMPT,
    "question": QUESTION_PROMPT,
    "custom": CUSTOM_PROMPT,
}

END_PROMPT = """Remember to always end your response with nothing but a float score.
Float score:"""
