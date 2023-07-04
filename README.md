# Auto-GPT Benchmark

A repo built for the purpose of benchmarking the performance of agents far and wide, regardless of how they are set up and how they work

## As a user

1. `pip install auto-gpt-benchmarks`
2. Add boilerplate code to run and kill agent
3. `agbenchmark start`
   - `--category challenge_category` to run tests in a specific category
   - `--mock` to only run mock tests if they exists for each test
   - `--noreg` to skip any tests that have passed in the past. When you run without this flag and a previous challenge that passed fails, it will now not be regression tests
4. We call boilerplate code for your agent
5. Show pass rate of tests, logs, and any other metrics

## Contributing

##### Diagrams: https://whimsical.com/agbenchmark-5n4hXBq1ZGzBwRsK4TVY7x

### To run the existing mocks

1. clone the repo `auto-gpt-benchmarks`
2. `pip install poetry`
3. `poetry shell`
4. `poetry install`
5. `cp .env_example .env`
6. `agbenchmark start --mock`
   Keep config the same and watch the logs :)

### To run with mini-agi

1. Navigate to `auto-gpt-benchmarks/agent/mini-agi`
2. `pip install -r requirements.txt`
3. `cp .env_example .env`, set `PROMPT_USER=false` and add your `OPENAI_API_KEY=`. Sset `MODEL="gpt-3.5-turbo"` if you don't have access to `gpt-4` yet. Also make sure you have Python 3.10^ installed
4. Make sure to follow the commands above, and remove mock flag `agbenchmark start`

- To add requirements `poetry add requirement`.

Feel free to create prs to merge with `main` at will (but also feel free to ask for review) - if you can't send msg in R&D chat for access.

If you push at any point and break things - it'll happen to everyone - fix it asap. Step 1 is to revert `master` to last working commit

Let people know what beautiful code you write does, document everything well

Share your progress :)

### Pytest

an example of a test is below, use it as a template and change the class name, the .json name, what the test depends on and it's name, and the scoring logic

```python
import pytest
from agbenchmark.tests.basic_abilities.BasicChallenge import BasicChallenge
import os


class TestWriteFile(BasicChallenge):
    """Testing if LLM can write to a file"""

    def get_file_path(self) -> str:  # all tests must implement this method
        return os.path.join(os.path.dirname(__file__), "w_file_data.json")

    @pytest.mark.depends(on=[], name="basic_write_file")
    def test_method(self, workspace):
        # implement scoring logic by looking at workspace
```

All challenges will inherit from parent class which has the mark and any specific methods for their category

```python
@pytest.mark.basic
class BasicChallenge(Challenge):
    pass
```

Add the below to create a file in the workspace prior to running a challenge. Only use when a file is needed to be created in the workspace prior to a test, such as with the read_file_test. 
```python
@pytest.fixture(
        scope="module", autouse=True
    )  # this is specific to setting up a file for the test, not all tests have this
    def setup_module(self, workspace):
        Challenge.write_to_file(
            workspace, self.data.ground.files[0], "this is how we're doing"
        )
```

#### The main Challenge class has all the parametrization and loading logic so that all tests can inherit from it. It lives within [this file](https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks/blob/master/agbenchmark/Challenge.py)

## Workspace

If `--mock` flag is used it is at `agbenchmark/mocks/workspace`. Otherwise for mini-agi it is at `C:/Users/<name>/miniagi` - it will be automitcally set on config

#### Dataset

Manually created, existing challenges within Auto-Gpt, https://osu-nlp-group.github.io/Mind2Web/

## Repo

```
|-- auto-gpt-benchmarks/ **main project directory**
| |-- metrics.py **combining scores, metrics, final evaluation**
| |-- start_benchmark.py **entry point from cli**
| |-- conftest.py **config, workspace creation + teardown, regression tesst markers, parameterization**
| |-- Challenge.py **easy challenge creation class**
| |-- config.json **workspace folder**
| |-- challenges/ **challenges across different domains**
| | |-- adaptability/
| | |-- basic_abilities/
| | |-- code/
| | |-- memory/
| | |-- retrieval/
| | |-- web_navigation/
| | |-- writing/
| |-- tests/
| | |-- basic_abilities/ **every llm should pass these challenges**
| | |-- regression/ **challenges that already passed**
```

## How to add new agents to agbenchmark ?
Example with smol developer.

1- Create a github branch with your agent following the same pattern as this example:

https://github.com/smol-ai/developer/pull/114/files

2- Create the submodule and the github workflow by following the same pattern as this example:

https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks/pull/48/files
