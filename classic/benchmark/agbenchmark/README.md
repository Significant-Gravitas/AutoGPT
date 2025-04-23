## As a user

1. `pip install auto-gpt-benchmarks`
2. Add boilerplate code to run and kill agent
3. `agbenchmark`
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
6. `git submodule update --init --remote --recursive`
7. `uvicorn server:app --reload`
8. `agbenchmark --mock`
   Keep config the same and watch the logs :)

### To run with mini-agi

1. Navigate to `auto-gpt-benchmarks/agent/mini-agi`
2. `pip install -r requirements.txt`
3. `cp .env_example .env`, set `PROMPT_USER=false` and add your `OPENAI_API_KEY=`. Sset `MODEL="gpt-3.5-turbo"` if you don't have access to `gpt-4` yet. Also make sure you have Python 3.10^ installed
4. set `AGENT_NAME=mini-agi` in `.env` file and where you want your `REPORTS_FOLDER` to be
5. Make sure to follow the commands above, and remove mock flag `agbenchmark`

- To add requirements `poetry add requirement`.

Feel free to create prs to merge with `main` at will (but also feel free to ask for review) - if you can't send msg in R&D chat for access.

If you push at any point and break things - it'll happen to everyone - fix it asap. Step 1 is to revert `master` to last working commit

Let people know what beautiful code you write does, document everything well

Share your progress :)

#### Dataset

Manually created, existing challenges within Auto-Gpt, https://osu-nlp-group.github.io/Mind2Web/

## How do I add new agents to agbenchmark ?

Example with smol developer.

1- Create a github branch with your agent following the same pattern as this example:

https://github.com/smol-ai/developer/pull/114/files

2- Create the submodule and the github workflow by following the same pattern as this example:

https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks/pull/48/files

## How do I run agent in different environments?

**To just use as the benchmark for your agent**. `pip install` the package and run `agbenchmark`

**For internal Auto-GPT ci runs**, specify the `AGENT_NAME` you want you use and set the `HOME_ENV`.
Ex. `AGENT_NAME=mini-agi`

**To develop agent alongside benchmark**, you can specify the `AGENT_NAME` you want you use and add as a submodule to the repo
