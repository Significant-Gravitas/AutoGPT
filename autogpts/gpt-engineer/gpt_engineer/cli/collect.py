"""
This module provides functionalities to handle and send learning data to RudderStack
for the purpose of analysis and to improve the gpt-engineer system. The data is sent
only when the user gives consent to share.

The module provides the following main functions:

- `send_learning`: Directly send a learning data to RudderStack.
- `collect_learnings`: Extract, possibly adjust, and send the learning data based on
  provided input parameters.
- `steps_file_hash`: Computes the SHA-256 hash of the steps file, which might be used
  for identifying the exact version or changes in the steps.

Dependencies:
- hashlib: For generating SHA-256 hash.
- typing: For type annotations.
- gpt_engineer.core: Core functionalities of gpt-engineer.
- gpt_engineer.cli.learning: Handles the extraction of learning data.

Note:
    Data sent to RudderStack is not shared with third parties and is used solely to
    improve gpt-engineer and allow it to handle a broader range of use cases.
    Consent logic is in gpt_engineer/learning.py.

"""
import hashlib

from typing import List

from gpt_engineer.core import steps
from gpt_engineer.core.db import DBs
from gpt_engineer.core.domain import Step
from gpt_engineer.cli.learning import Learning, extract_learning

...

import hashlib

from typing import List

from gpt_engineer.core import steps
from gpt_engineer.core.db import DBs
from gpt_engineer.core.domain import Step
from gpt_engineer.cli.learning import Learning, extract_learning


def send_learning(learning: Learning):
    """
    Send the learning data to RudderStack for analysis.

    Note:
    This function is only called if consent is given to share data.
    Data is not shared to a third party. It is used with the sole purpose of
    improving gpt-engineer, and letting it handle more use cases.
    Consent logic is in gpt_engineer/learning.py

    Parameters
    ----------
    learning : Learning
        The learning data to send.
    """
    import rudderstack.analytics as rudder_analytics

    rudder_analytics.write_key = "2Re4kqwL61GDp7S8ewe6K5dbogG"
    rudder_analytics.dataPlaneUrl = "https://gptengineerezm.dataplane.rudderstack.com"

    rudder_analytics.track(
        user_id=learning.session,
        event="learning",
        properties=learning.to_dict(),  # type: ignore
    )


def collect_learnings(model: str, temperature: float, steps: List[Step], dbs: DBs):
    """
    Collect the learning data and send it to RudderStack for analysis.

    Parameters
    ----------
    model : str
        The name of the model used.
    temperature : float
        The temperature used.
    steps : List[Step]
        The list of steps.
    dbs : DBs
        The database containing the workspace.
    """
    learnings = extract_learning(
        model, temperature, steps, dbs, steps_file_hash=steps_file_hash()
    )
    try:
        send_learning(learnings)
    except RuntimeError as e:
        # try to remove some parts of learning that might be too big
        # rudderstack max event size is 32kb
        overflow = len(learnings.to_json()) - (32 << 10)  # type: ignore
        assert overflow > 0, f"encountered error {e} but overflow is {overflow}"

        learnings.logs = (
            learnings.logs[: -overflow - 200] + f"\n\n[REMOVED {overflow} CHARACTERS]"
        )
        print(
            "WARNING: learning too big, removing some parts. "
            "Please report if this results in a crash."
        )
        send_learning(learnings)


def steps_file_hash():
    """
    Compute the SHA-256 hash of the steps file.

    Returns
    -------
    str
        The SHA-256 hash of the steps file.
    """
    with open(steps.__file__, "r") as f:
        content = f.read()
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
