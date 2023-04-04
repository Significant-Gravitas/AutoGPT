import json
import subprocess
from tempfile import NamedTemporaryFile

import pytest

from openai.datalib import (
    HAS_NUMPY,
    HAS_PANDAS,
    NUMPY_INSTRUCTIONS,
    PANDAS_INSTRUCTIONS,
)


@pytest.mark.skipif(not HAS_PANDAS, reason=PANDAS_INSTRUCTIONS)
@pytest.mark.skipif(not HAS_NUMPY, reason=NUMPY_INSTRUCTIONS)
def test_long_examples_validator() -> None:
    """
    Ensures that long_examples_validator() handles previously applied recommendations,
    namely dropped duplicates, without resulting in a KeyError.
    """

    # data
    short_prompt = "a prompt "
    long_prompt = short_prompt * 500

    short_completion = "a completion "
    long_completion = short_completion * 500

    # the order of these matters
    unprepared_training_data = [
        {"prompt": long_prompt, "completion": long_completion},  # 1 of 2 duplicates
        {"prompt": short_prompt, "completion": short_completion},
        {"prompt": long_prompt, "completion": long_completion},  # 2 of 2 duplicates
    ]

    with NamedTemporaryFile(suffix=".jsonl", mode="w") as training_data:
        print(training_data.name)
        for prompt_completion_row in unprepared_training_data:
            training_data.write(json.dumps(prompt_completion_row) + "\n")
            training_data.flush()

        prepared_data_cmd_output = subprocess.run(
            [f"openai tools fine_tunes.prepare_data -f {training_data.name}"],
            stdout=subprocess.PIPE,
            text=True,
            input="y\ny\ny\ny\ny",  # apply all recommendations, one at a time
            stderr=subprocess.PIPE,
            encoding="utf-8",
            shell=True,
        )

    # validate data was prepared successfully
    assert prepared_data_cmd_output.stderr == ""
    # validate get_long_indexes() applied during optional_fn() call in long_examples_validator()
    assert "indices of the long examples has changed" in prepared_data_cmd_output.stdout
    
    return prepared_data_cmd_output.stdout
