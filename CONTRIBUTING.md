# Contributing to Auto-GPT

First of all, thank you for considering contributing to our project! We appreciate your time and effort, and we value any contribution, whether it's reporting a bug, suggesting a new feature, or submitting a pull request.

This document provides guidelines and best practices to help you contribute effectively.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct]. Please read it to understand the expectations we have for everyone who contributes to this project.

[Code of Conduct]: https://significant-gravitas.github.io/Auto-GPT/code-of-conduct.md

## 📢 A Quick Word
Right now we will not be accepting any Contributions that add non-essential commands to Auto-GPT.

However, you absolutely can still add these commands to Auto-GPT in the form of plugins.
Please check out this [template](https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template).

## Getting Started

1. Fork the repository and clone your fork.
2. Create a new branch for your changes (use a descriptive name, such as `fix-bug-123` or `add-new-feature`).
3. Make your changes in the new branch.
4. Test your changes thoroughly.
5. Commit and push your changes to your fork.
6. Create a pull request following the guidelines in the [Submitting Pull Requests](#submitting-pull-requests) section.

## How to Contribute

### Reporting Bugs

If you find a bug in the project, please create an issue on GitHub with the following information:

- A clear, descriptive title for the issue.
- A description of the problem, including steps to reproduce the issue.
- Any relevant logs, screenshots, or other supporting information.

### Suggesting Enhancements

If you have an idea for a new feature or improvement, please create an issue on GitHub with the following information:

- A clear, descriptive title for the issue.
- A detailed description of the proposed enhancement, including any benefits and potential drawbacks.
- Any relevant examples, mockups, or supporting information.

### Submitting Pull Requests

When submitting a pull request, please ensure that your changes meet the following criteria:

- Your pull request should be atomic and focus on a single change.
- Your pull request should include tests for your change. We automatically enforce this with [CodeCov](https://docs.codecov.com/docs/commit-status)
- You should have thoroughly tested your changes with multiple different prompts.
- You should have considered potential risks and mitigations for your changes.
- You should have documented your changes clearly and comprehensively.
- You should not include any unrelated or "extra" small tweaks or changes.

## Style Guidelines

### Code Formatting

We use the `black` and `isort` code formatters to maintain a consistent coding style across the project. Please ensure that your code is formatted properly before submitting a pull request.

To format your code, run the following commands in the project's root directory:

```bash
python -m black .
python -m isort .
```

Or if you have these tools installed globally:
```bash
black .
isort .
```

### Pre-Commit Hooks

We use pre-commit hooks to ensure that code formatting and other checks are performed automatically before each commit. To set up pre-commit hooks for this project, follow these steps:

Install the pre-commit package using pip:
```bash
pip install pre-commit
```

Run the following command in the project's root directory to install the pre-commit hooks:
```bash
pre-commit install
```

Now, the pre-commit hooks will run automatically before each commit, checking your code formatting and other requirements.

If you encounter any issues or have questions, feel free to reach out to the maintainers or open a new issue on GitHub. We're here to help and appreciate your efforts to contribute to the project.

Happy coding, and once again, thank you for your contributions!

Maintainers will look at PR that have no merge conflicts when deciding what to add to the project. Make sure your PR shows up here:
https://github.com/Significant-Gravitas/Auto-GPT/pulls?q=is%3Apr+is%3Aopen+-label%3Aconflicts

## Testing your changes

If you add or change code, make sure the updated code is covered by tests.
To increase coverage if necessary, [write tests using pytest].

For more info on running tests, please refer to ["Running tests"](https://significant-gravitas.github.io/Auto-GPT/testing/).

[write tests using pytest]: https://realpython.com/pytest-python-testing/

### API-dependent tests

To run tests that involve making calls to the OpenAI API, we use VCRpy. It caches known
requests and matching responses in so-called *cassettes*, allowing us to run the tests
in CI without needing actual API access.

When changes cause a test prompt to be generated differently, it will likely miss the
cache and make a request to the API, updating the cassette with the new request+response.
*Be sure to include the updated cassette in your PR!*

When you run Pytest locally:

- If no prompt change: you will not consume API tokens because there are no new OpenAI calls required.
- If the prompt changes in a way that the cassettes are not reusable:
    - If no API key, the test fails. It requires a new cassette. So, add an API key to .env.
    - If the API key is present, the tests will make a real call to OpenAI.
        - If the test ends up being successful, your prompt changes didn't introduce regressions. This is good. Commit your cassettes to your PR.
        - If the test is unsuccessful:
            - Either: Your change made Auto-GPT less capable, in that case, you have to change your code.
            - Or: The test might be poorly written. In that case, you can make suggestions to change the test.

In our CI pipeline, Pytest will use the cassettes and not call paid API providers, so we need your help to record the replays that you break.


### Community Challenges
Challenges are goals we need Auto-GPT to achieve. 
To pick the challenge you like, go to the tests/integration/challenges folder and select the areas you would like to work on. 
- a challenge is new if level_currently_beaten is None
- a challenge is in progress if level_currently_beaten is greater or equal to 1
- a challenge is beaten if level_currently_beaten = max_level

Here is an example of how to run the memory challenge A and attempt to beat level 3.

pytest -s tests/integration/challenges/memory/test_memory_challenge_a.py --level=3

To beat a challenge, you're not allowed to change anything in the tests folder, you have to add code in the autogpt folder

Challenges use cassettes. Cassettes allow us to replay your runs in our CI pipeline.
Don't hesitate to delete the cassettes associated to the challenge you're working on if you need to. Otherwise it will keep replaying the last run.

Once you've beaten a new level of a challenge, please create a pull request and we will analyze how you changed Auto-GPT to beat the challenge.
