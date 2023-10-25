"""
This module provides tools and data structures for supporting a feedback loop in the GPT Engineer application.

The primary intent of this module is to gather feedback from the user on the output of the gpt-engineer tool,
with their consent, and to store this feedback for further analysis and improvement of the tool.

Classes:
----------
Review:
    Represents user's review of the generated code.
Learning:
    Represents the metadata and feedback collected for a session in which gpt-engineer was used.

Functions:
----------
human_review_input() -> Review:
    Interactively gathers feedback from the user regarding the performance of generated code.

check_consent() -> bool:
    Checks if the user has previously given consent to store their data and if not, asks for it.

collect_consent() -> bool:
    Verifies if the user has given consent to store their data or prompts for it.

ask_if_can_store() -> bool:
    Asks the user if it's permissible to store their data for gpt-engineer improvement.

logs_to_string(steps: List[Step], logs: DB) -> str:
    Converts logs of steps into a readable string format.

extract_learning(model: str, temperature: float, steps: List[Step], dbs: DBs, steps_file_hash) -> Learning:
    Extracts feedback and session details to create a Learning instance.

get_session() -> str:
    Retrieves a unique identifier for the current user session.

Constants:
----------
TERM_CHOICES:
    Terminal color choices for user interactive prompts.
"""

import json
import random
import tempfile

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dataclasses_json import dataclass_json
from termcolor import colored

from gpt_engineer.core.db import DB, DBs
from gpt_engineer.core.domain import Step


@dataclass_json
@dataclass
class Review:
    ran: Optional[bool]
    perfect: Optional[bool]
    works: Optional[bool]
    comments: str
    raw: str


@dataclass_json
@dataclass
class Learning:
    model: str
    temperature: float
    steps: str
    steps_file_hash: str
    prompt: str
    logs: str
    workspace: str
    feedback: Optional[str]
    session: str
    review: Optional[Review]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "0.3"


TERM_CHOICES = (
    colored("y", "green")
    + "/"
    + colored("n", "red")
    + "/"
    + colored("u", "yellow")
    + "(ncertain): "
)


def human_review_input() -> Review:
    """
    Ask the user to review the generated code and return their review.

    Returns
    -------
    Review
        The user's review of the generated code.
    """
    print()
    if not check_consent():
        return None
    print()
    print(
        colored("To help gpt-engineer learn, please answer 3 questions:", "light_green")
    )
    print()

    ran = input("Did the generated code run at all? " + TERM_CHOICES)
    while ran not in ("y", "n", "u"):
        ran = input("Invalid input. Please enter y, n, or u: ")

    perfect = ""
    useful = ""

    if ran == "y":
        perfect = input(
            "Did the generated code do everything you wanted? " + TERM_CHOICES
        )
        while perfect not in ("y", "n", "u"):
            perfect = input("Invalid input. Please enter y, n, or u: ")

        if perfect != "y":
            useful = input("Did the generated code do anything useful? " + TERM_CHOICES)
            while useful not in ("y", "n", "u"):
                useful = input("Invalid input. Please enter y, n, or u: ")

    comments = ""
    if perfect != "y":
        comments = input(
            "If you have time, please explain what was not working "
            + colored("(ok to leave blank)\n", "light_green")
        )

    return Review(
        raw=", ".join([ran, perfect, useful]),
        ran={"y": True, "n": False, "u": None, "": None}[ran],
        works={"y": True, "n": False, "u": None, "": None}[useful],
        perfect={"y": True, "n": False, "u": None, "": None}[perfect],
        comments=comments,
    )


def check_consent() -> bool:
    """
    Check if the user has given consent to store their data.
    If not, ask for their consent.
    """
    path = Path(".gpte_consent")
    if path.exists() and path.read_text() == "true":
        return True
    answer = input("Is it ok if we store your prompts to learn? (y/n)")
    while answer.lower() not in ("y", "n"):
        answer = input("Invalid input. Please enter y or n: ")

    if answer.lower() == "y":
        path.write_text("true")
        print(colored("Thank you️", "light_green"))
        print()
        print("(If you change your mind, delete the file .gpte_consent)")
        return True
    else:
        print(colored("We understand ❤️", "light_green"))
        return False


def collect_consent() -> bool:
    """
    Check if the user has given consent to store their data.
    If not, ask for their consent.

    Returns
    -------
    bool
        True if the user has given consent, False otherwise.
    """
    consent_flag = Path(".gpte_consent")
    if consent_flag.exists():
        return consent_flag.read_text() == "true"

    if ask_if_can_store():
        consent_flag.write_text("true")
        print()
        print("(If you change your mind, delete the file .gpte_consent)")
        return True
    return False


def ask_if_can_store() -> bool:
    """
    Ask the user if their data can be stored.

    Returns
    -------
    bool
        True if the user agrees to have their data stored, False otherwise.
    """
    print()
    can_store = input(
        "Have you understood and agree to that "
        + colored("OpenAI ", "light_green")
        + "and "
        + colored("gpt-engineer ", "light_green")
        + "store anonymous learnings about how gpt-engineer is used "
        + "(with the sole purpose of improving it)?\n(y/n)"
    ).lower()
    while can_store not in ("y", "n"):
        can_store = input("Invalid input. Please enter y or n: ").lower()

    if can_store == "n":
        print(colored("Ok we understand", "light_green"))

    return can_store == "y"


def logs_to_string(steps: List[Step], logs: DB) -> str:
    """
    Convert the logs of the steps to a string.

    Parameters
    ----------
    steps : List[Step]
        The list of steps.
    logs : DB
        The database containing the logs.

    Returns
    -------
    str
        The logs of the steps as a string.
    """
    chunks = []
    for step in steps:
        chunks.append(f"--- {step.__name__} ---\n")
        chunks.append(logs[step.__name__])
    return "\n".join(chunks)


def extract_learning(
    model: str, temperature: float, steps: List[Step], dbs: DBs, steps_file_hash
) -> Learning:
    """
    Extract the learning data from the steps and databases.

    Parameters
    ----------
    model : str
        The name of the model used.
    temperature : float
        The temperature used.
    steps : List[Step]
        The list of steps.
    dbs : DBs
        The databases containing the input, logs, memory, and workspace.
    steps_file_hash : str
        The hash of the steps file.

    Returns
    -------
    Learning
        The extracted learning data.
    """
    review = None
    if "review" in dbs.memory:
        review = Review.from_json(dbs.memory["review"])  # type: ignore
    learning = Learning(
        prompt=dbs.input["prompt"],
        model=model,
        temperature=temperature,
        steps=json.dumps([step.__name__ for step in steps]),
        steps_file_hash=steps_file_hash,
        feedback=dbs.input.get("feedback"),
        session=get_session(),
        logs=logs_to_string(steps, dbs.logs),
        workspace=dbs.memory.get("all_output.txt"),
        review=review,
    )
    return learning


def get_session() -> str:
    """
    Returns a unique user id for the current user project (session).

    Returns
    -------
    str
        The unique user id.
    """
    path = Path(tempfile.gettempdir()) / "gpt_engineer_user_id.txt"

    try:
        if path.exists():
            user_id = path.read_text()
        else:
            # random uuid:
            user_id = str(random.randint(0, 2**32))
            path.write_text(user_id)
        return user_id
    except IOError:
        return "ephemeral_" + str(random.randint(0, 2**32))
