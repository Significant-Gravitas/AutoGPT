"""
This module provides annotations related to the steps workflow in GPT Engineer.

Imports:
    - `Callable`, `List`, and `TypeVar` from `typing` module, which are used for type hinting.
    - `AI` class from the `gpt_engineer.core.ai` module.
    - `DBs` class from the `gpt_engineer.core.db` module.

Variables:
    - `Step`: This is a generic type variable that represents a Callable or function type. The function
      is expected to accept two parameters: an instance of `AI` and an instance of `DBs`. The function
      is expected to return a list of dictionaries.
"""

from typing import Callable, List, TypeVar

from gpt_engineer.core.ai import AI
from gpt_engineer.core.db import DBs

Step = TypeVar("Step", bound=Callable[[AI, DBs], List[dict]])
