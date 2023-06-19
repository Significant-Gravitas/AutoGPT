# agbenchmark

A repo built for the purpose of benchmarking the performance of agents far and wide, regardless of how they are set up and how they work

Simple boilerplate code that spins up a webserver to plug their agent into. We call multiple tasks by invoking different pytest commands on folders and once the agent stops or reaches 50 loops (which they will have to define). We handle the deletion of files after a run loop ends. Then we call call the POST request for the next task. Then we will spit out a combined benchmark once all tests run

- Agent adds tests by adding to our repo
- Agent abstracted from benchmark
- Scalable (parallel servers running tests)
- Better standardization

##### Diagrams (out of date, cloud oriented): https://whimsical.com/agbenchmark-5n4hXBq1ZGzBwRsK4TVY7x

## Contributing

- Make sure you have `poetry` installed - `pip install poetry`.
- Then `poetry install` for dependencies

- To add requirements `poetry add requirement`.
- To run in venv `poetry run python script.py`

Feel free to merge with `main` at will (but also to ask for review) - if you can't send msg in R&D chat for access.

If you push at any point and break things - it'll happen to everyone - fix it asap. Step 1 is to revert `main` to last working commit

Let people know what beautiful code you write does, document everything well

Share your progress :)

## Api

FastAPI with REST, import requests

```
POST hostname:8080/challenges
{
   "test_name": ""
   "challenge": "memory" - optional
}
```

## Auth:

get preSignedUrl from API

```
POST preSignedUrl
{
   "artifacts": [{}]
}
```

## Workspace

Kubernetes with AWS3 or GCP

## Challenges

#### Dataset

Manually created, existing challenges within Auto-Gpt, https://osu-nlp-group.github.io/Mind2Web/

#### Simple challenge creation through a DSL (domain specific language)

```
Challenge TicTacToeCoding
    Description "The agent should implement a basic tic-tac-toe game in Python."
    Artifacts {
        Code "tictactoe.py"
    }
    Tasks {
        Code "Write a function to initialize the game board."
        Code "Write a function to handle a player's turn."
        Code "Write a function to check for a winning move."
        Test "Write tests for the blog post model, serializer, and view."
        Command "Run Django's test suite to ensure everything is working as expected."
    }
    SuccessCriteria {
        Correctness "The game should correctly alternate between two players."
        Correctness "The game should correctly identify a winning move."
        Efficiency "The game should not use unnecessary computational resources."
        Design "The solution should follow good practices for Django and Django Rest Framework."
    }
EndChallenge
```

#### Validators

Designed to handle specific types of output (e.g., text, code, structured data)

#### Logging

Log different requests coming in - write file, change file, etc. Maybe a db in the future for metrics, logs, etc

#### Written Challenges

For code, writing we can create a reference text and use metrics like METEOR, BERTScore, BARTScore

## Repo

```
|-- agbenchmark/ **main project directory**
| |-- **init**.py
| |-- server/
| | |-- **init**.py
| | |-- api.py **opens server on host and exposes urls**
| | |-- utils.py
| |-- benchmark/
| | |-- **init**.py
| | |-- benchmark.py **combining scores, metrics, final evaluation**
| | |-- run.py **entry point. sets everything up**
| | |-- challenges/ **challenges across different metrics**
| | | |-- **init**.py
| | | |-- Challenge.py **easy challenge creation through Challenge class. potentially how DSL is defined. may need to inherit challenge class like Adaptability(Challenge)**
| | | |-- utils.py
| | | |-- adaptability.py
| | | |-- basic_abilities.py
| | | |-- code.py
| | | |-- memory.py
| | | |-- retrieval.py
| | | |-- web_navigation.py
| | | |-- writing.py
| |-- workspace/ **workspace related func**
| | |-- **init**.py
| | |-- workspace_manager.py **creation, deletion, preSignedUrl generation**
| | |-- cloud_services/
| | | |-- **init**.py
| | | |-- aws.py **not finalized, but write, read, and del files**
|-- tests/ **test func of agbenchmark**
| |-- **init**.py
| |-- test_api.py
| |-- test_benchmark.py
| |-- test_workspace_manager.py
```

Later: GitHub Actions integration, OpenAPI?, good versioning and backward compatibility
