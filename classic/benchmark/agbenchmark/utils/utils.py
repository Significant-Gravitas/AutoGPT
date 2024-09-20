# radio charts, logs, helper functions for tests, anything else relevant.
import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar, overload

import click
from dotenv import load_dotenv
from pydantic import BaseModel

from agbenchmark.reports.processing.report_types import Test
from agbenchmark.utils.data_types import DIFFICULTY_MAP, DifficultyLevel

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME")

logger = logging.getLogger(__name__)

T = TypeVar("T")
E = TypeVar("E", bound=Enum)


def replace_backslash(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(
            r"\\+", "/", value
        )  # replace one or more backslashes with a forward slash
    elif isinstance(value, list):
        return [replace_backslash(i) for i in value]
    elif isinstance(value, dict):
        return {k: replace_backslash(v) for k, v in value.items()}
    else:
        return value


def get_test_path(json_file: str | Path) -> str:
    if isinstance(json_file, str):
        json_file = Path(json_file)

    # Find the index of "agbenchmark" in the path parts
    try:
        agbenchmark_index = json_file.parts.index("benchmark")
    except ValueError:
        raise ValueError("Invalid challenge location.")

    # Create the path from "agbenchmark" onwards
    challenge_location = Path(*json_file.parts[agbenchmark_index:])

    formatted_location = replace_backslash(str(challenge_location))
    if isinstance(formatted_location, str):
        return formatted_location
    else:
        return str(challenge_location)


def get_highest_success_difficulty(
    data: dict[str, Test], just_string: Optional[bool] = None
) -> str:
    highest_difficulty = None
    highest_difficulty_level = 0

    for test_name, test_data in data.items():
        try:
            if any(r.success for r in test_data.results):
                difficulty_str = test_data.difficulty
                if not difficulty_str:
                    continue

                try:
                    difficulty_enum = DifficultyLevel[difficulty_str.lower()]
                    difficulty_level = DIFFICULTY_MAP[difficulty_enum]

                    if difficulty_level > highest_difficulty_level:
                        highest_difficulty = difficulty_enum
                        highest_difficulty_level = difficulty_level
                except KeyError:
                    logger.warning(
                        f"Unexpected difficulty level '{difficulty_str}' "
                        f"in test '{test_name}'"
                    )
                    continue
        except Exception as e:
            logger.warning(
                "An unexpected error [1] occurred while analyzing report [2]."
                "Please notify a maintainer.\n"
                f"Report data [1]: {data}\n"
                f"Error [2]: {e}"
            )
            logger.warning(
                "Make sure you selected the right test, no reports were generated."
            )
            break

    if highest_difficulty is not None:
        highest_difficulty_str = highest_difficulty.name  # convert enum to string
    else:
        highest_difficulty_str = ""

    if highest_difficulty_level and not just_string:
        return f"{highest_difficulty_str}: {highest_difficulty_level}"
    elif highest_difficulty_str:
        return highest_difficulty_str
    return "No successful tests"


# def get_git_commit_sha(directory: Path) -> Optional[str]:
#     try:
#         repo = git.Repo(directory)
#         remote_url = repo.remotes.origin.url
#         if remote_url.endswith(".git"):
#             remote_url = remote_url[:-4]
#         git_commit_sha = f"{remote_url}/tree/{repo.head.commit.hexsha}"

#         # logger.debug(f"GIT_COMMIT_SHA: {git_commit_sha}")
#         return git_commit_sha
#     except Exception:
#         # logger.error(f"{directory} is not a git repository!")
#         return None


def write_pretty_json(data, json_file):
    sorted_data = deep_sort(data)
    json_graph = json.dumps(sorted_data, indent=4)
    with open(json_file, "w") as f:
        f.write(json_graph)
        f.write("\n")


def pretty_print_model(model: BaseModel, include_header: bool = True) -> None:
    indent = ""
    if include_header:
        # Try to find the ID and/or name attribute of the model
        id, name = None, None
        for attr, value in model.model_dump().items():
            if attr == "id" or attr.endswith("_id"):
                id = value
            if attr.endswith("name"):
                name = value
            if id and name:
                break
        identifiers = [v for v in [name, id] if v]
        click.echo(
            f"{model.__repr_name__()}{repr(identifiers) if identifiers else ''}:"
        )
        indent = " " * 2

    k_col_width = max(len(k) for k in model.model_dump().keys())
    for k, v in model.model_dump().items():
        v_fmt = repr(v)
        if v is None or v == "":
            v_fmt = click.style(v_fmt, fg="black")
        elif type(v) is bool:
            v_fmt = click.style(v_fmt, fg="green" if v else "red")
        elif type(v) is str and "\n" in v:
            v_fmt = f"\n{v}".replace(
                "\n", f"\n{indent} {click.style('|', fg='black')} "
            )
        if isinstance(v, Enum):
            v_fmt = click.style(v.value, fg="blue")
        elif type(v) is list and len(v) > 0 and isinstance(v[0], Enum):
            v_fmt = ", ".join(click.style(lv.value, fg="blue") for lv in v)
        click.echo(f"{indent}{k: <{k_col_width}}  = {v_fmt}")


def deep_sort(obj):
    """
    Recursively sort the keys in JSON object
    """
    if isinstance(obj, dict):
        return {k: deep_sort(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [deep_sort(elem) for elem in obj]
    return obj


@overload
def sorted_by_enum_index(
    sortable: Iterable[E],
    enum: type[E],
    *,
    reverse: bool = False,
) -> list[E]:
    ...


@overload
def sorted_by_enum_index(
    sortable: Iterable[T],
    enum: type[Enum],
    *,
    key: Callable[[T], Enum | None],
    reverse: bool = False,
) -> list[T]:
    ...


def sorted_by_enum_index(
    sortable: Iterable[T],
    enum: type[Enum],
    *,
    key: Optional[Callable[[T], Enum | None]] = None,
    reverse: bool = False,
) -> list[T]:
    return sorted(
        sortable,
        key=lambda x: (
            enum._member_names_.index(e.name)  # type: ignore
            if (e := key(x) if key else x)
            else 420e3
        ),
        reverse=reverse,
    )
