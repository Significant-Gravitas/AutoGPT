# Contributing to ProjectName

First of all, thank you for considering contributing to our project! We appreciate your time and effort, and we value any contribution, whether it's reporting a bug, suggesting a new feature, or submitting a pull request.

This document provides guidelines and best practices to help you contribute effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Style Guidelines](#style-guidelines)
  - [Code Formatting](#code-formatting)
  - [Pre-Commit Hooks](#pre-commit-hooks)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand the expectations we have for everyone who contributes to this project.

## 📢 A Quick Word
Right now we will not be accepting any Contributions that add non-essential commands to Auto-GPT.

However, you absolutely can still add these commands to Auto-GPT in the form of plugins. Please check out this [template](https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template).
> ⚠️ Plugin support is expected to ship within the week. You can follow PR #757 for more updates!

## Getting Started

To start contributing, follow these steps:

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
- Your pull request should include tests for your change.
- You should have thoroughly tested your changes with multiple different prompts.
- You should have considered potential risks and mitigations for your changes.
- You should have documented your changes clearly and comprehensively.
- You should not include any unrelated or "extra" small tweaks or changes.

## Style Guidelines

### Code Formatting

We use the `black` code formatter to maintain a consistent coding style across the project. Please ensure that your code is formatted using `black` before submitting a pull request. You can install `black` using `pip`:

```bash
pip install black
```

To format your code, run the following command in the project's root directory:

```bash
black .
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

https://github.com/Torantulino/Auto-GPT/pulls?q=is%3Apr+is%3Aopen+-is%3Aconflict+