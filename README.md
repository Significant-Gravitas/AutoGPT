# Auto-GPT Benchmark

A repo built for the purpose of benchmarking the performance of agents far and wide, regardless of how they are set up and how they work

##### Diagrams: https://whimsical.com/agbenchmark-5n4hXBq1ZGzBwRsK4TVY7x

## Contributing

- Make sure you have `poetry` installed - `pip install poetry`.
- Then `poetry install` for dependencies

- To add requirements `poetry add requirement`.
- To run in venv `poetry run python script.py`

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

```
@pytest.mark.parametrize(
"server_response",
["VARIABLE"], # VARIABLE = the query/goal you provide to the model
indirect=True,
)
@pytest.mark.(VARIABLE) # VARIABLE = category of the test
def test_file_in_workspace(workspace): # VARIABLE = the actual test that asserts
assert os.path.exists(os.path.join(workspace, "file_to_check.txt"))
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
