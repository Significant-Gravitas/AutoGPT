# Auto-GPT Benchmark

A repo built for the purpose of benchmarking the performance of agents far and wide, regardless of how they are set up and how they work

##### Diagrams: https://whimsical.com/agbenchmark-5n4hXBq1ZGzBwRsK4TVY7x

### To run the basic existing mock (June 21)

1. clone the repo `auto-gpt-benchmarks`
2. `pip install poetry`
3. `poetry shell`
4. `poetry install`
5. `agbenchmark start`
   Keep config the same and watch the logs :)

- To add requirements `poetry add requirement`.

Feel free to create prs to merge with `main` at will (but also feel free to ask for review) - if you can't send msg in R&D chat for access.

If you push at any point and break things - it'll happen to everyone - fix it asap. Step 1 is to revert `main` to last working commit

Let people know what beautiful code you write does, document everything well

Share your progress :)

## How this works

1. `pip install auto-gpt-benchmarks`
2. Add boilerplate code to start webserver to your agent (run loop and stop condition)
3. `agbenchmark start --category challenge_category` remove challenge flag to run all tests. specify config of hostname, port, and workspace directory
4. We call the server to run the agent for each test
5. Show pass rate of tests, logs, and any other metrics

### To run the basic existing mock (June 21)

1. clone the repo `auto-gpt-benchmarks`
2. `pip install poetry`
3. `poetry shell`
4. `poetry install`
5. `agbenchmark start`
   Keep config the same and watch the logs :)

#### Bonuses

- You can adds tests by git cloning auto-gpt-benchmarks to your repo
- Agent is abstracted from benchmark, don't need to do any extra setup other then starting the server
- Simple, easy to use
- Don't have to deal with cloud or parallelization yet

### Pytest

to create a test:

```python
import pytest
from agbenchmark.challenges.define_task_types import ChallengeData
from ..CategoryChallenge import CategoryChallenge
import os

data = ChallengeData.deserialize(
    os.path.join(os.path.dirname(__file__), "r_file_data.json")
)

class TestSomething(CategoryChallenge):
    """Testing if LLM can read a file"""

    @pytest.mark.parametrize(
        "run_agent",
        [(data.task, data.mock_func)],
        indirect=True,
    )
    def test_retrieval(
        self, workspace
    ):
        # scoring logic goes here
```

All challenges will inherit from parent class which has the mark

```python
@pytest.mark.basic
class BasicChallenge(Challenge):
    pass
```

If you want to add a custom mark to a Challenge, you must specify it before the test definition

```python
@pytest.mark.other_mark
def test_retrieval(self, workspace):
```

To add a dependency to a challenge use the following

```python
# to defining what a test depends on
from pytest_dependency import depends

def test1(self, request, workspace):
   depends(request, data.dependencies)
# for defining a test as a dependency
@pytest.mark.dependency()
def test2
```

Ordering of challenges needs to be used in combination with the above to make sure it executes afterwards

```python
@pytest.mark.run(order=1)
```

To create a file to test a challenge, add this to the challenge file which will create a file before running the server

```python
@pytest.fixture(scope="module", autouse=True)
def setup_module(workspace):
    if data.ground.should_contain:
        Challenge.write_to_file(
            workspace, data.ground.files[0], "this is how we're doing"
        )
```

## Api

FastAPI with REST, import requests to call in auto-gpt-benchmarks. Boilerplate code given to agent project to start server

## Workspace

Defined by the user on config

#### Dataset

Manually created, existing challenges within Auto-Gpt, https://osu-nlp-group.github.io/Mind2Web/

## Repo

```
|-- auto-gpt-benchmarks/ **main project directory**
| |-- metrics.py **combining scores, metrics, final evaluation**
| |-- start_benchmark.py **entry point from cli**
| |-- conftest.py **shared fixtures across all tests**
| |-- Challenge.py **easy challenge creation class?**
| |-- config.json **hostname, port, workspace folder**
| |-- challenges/ **challenges across different domains**
| | |-- adaptability/
| | |-- basic_abilities/
| | |-- code/
| | |-- memory/
| | |-- retrieval/
| | |-- web_navigation/
| | |-- writing/
| |-- tests/ **challenges across different metrics**
| | |-- basic_abilities/
| | |-- interface/
| |-- workspace/ **workspace related func**
| | |-- **init**.py
| | |-- workspace_manager.py **creation, deletion**
```

### Easy Challenge Creation

tbd, but potentially shared Challenge class that challenges instantiate as challenges need different utils/metrics for eval

#### Written Challenges

For code, writing we can create a reference text and use metrics like METEOR, BERTScore, BARTScore

#### Validators

Designed to handle specific types of output (e.g., text, code, structured data)

#### Logging

Log different requests coming in - write file, change file, etc. Maybe a db in the future for metrics, logs, etc

Later: GitHub Actions integration, OpenAPI?, good versioning and backward compatibility
