# Contributing to GPT Engineer

By participating in this project, you agree to abide by the [code of conduct](CODE_OF_CONDUCT.md).

## Getting Started

To get started with contributing, please follow these steps:

1. Fork the repository and clone it to your local machine.
2. Install any necessary dependencies.
3. Create a new branch for your changes: `git checkout -b my-branch-name`.
4. Make your desired changes or additions.
5. Run the tests to ensure everything is working as expected.
6. Commit your changes: `git commit -m "Descriptive commit message"`.
7. Push to the branch: `git push origin my-branch-name`.
8. Submit a pull request to the `main` branch of the original repository.

## Code Style

Please make sure to follow the established code style guidelines for this project. Consistent code style helps maintain readability and makes it easier for others to contribute to the project.

To enforce this we use [`pre-commit`](https://pre-commit.com/) to run [`black`](https://black.readthedocs.io/en/stable/index.html) and [`ruff`](https://beta.ruff.rs/docs/) on every commit.

`pre-commit` is part of our `requirements.txt` file so you should already have it installed. If you don't, you can install the library via pip with:

```bash
$ pip install -e .

# For docs building, install doc dependencies too

$ pip install -e .[doc]

# And then install the `pre-commit` hooks with:

$ pre-commit install

# output:
pre-commit installed at .git/hooks/pre-commit
```

Or you could just run `make dev-install` to install the dependencies and the hooks.

If you are not familiar with the concept of [git hooks](https://git-scm.com/docs/githooks) and/or [`pre-commit`](https://pre-commit.com/) please read the documentation to understand how they work.

As an introduction of the actual workflow, here is an example of the process you will encounter when you make a commit:

Let's add a file we have modified with some errors, see how the pre-commit hooks run `black` and fails.
`black` is set to automatically fix the issues it finds:

```bash
$ git add chat_to_files.py
$ git commit -m "commit message"
black....................................................................Failed
- hook id: black
- files were modified by this hook

reformatted chat_to_files.py

All done! ✨ 🍰 ✨
1 file reformatted.
```

You can see that `chat_to_files.py` is both staged and not staged for commit. This is because `black` has formatted it and now it is different from the version you have in your working directory. To fix this you can simply run `git add chat_to_files.py` again and now you can commit your changes.

```bash
$ git status
On branch pre-commit-setup
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
    modified:   chat_to_files.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
    modified:   chat_to_files.py
```

Now let's add the file again to include the latest commits and see how `ruff` fails.

```bash
$ git add chat_to_files.py
$ git commit -m "commit message"
black....................................................................Passed
ruff.....................................................................Failed
- hook id: ruff
- exit code: 1
- files were modified by this hook

Found 2 errors (2 fixed, 0 remaining).
```

Same as before, you can see that `chat_to_files.py` is both staged and not staged for commit. This is because `ruff` has formatted it and now it is different from the version you have in your working directory. To fix this you can simply run `git add chat_to_files.py` again and now you can commit your changes.

```bash
$ git add chat_to_files.py
$ git commit -m "commit message"
black....................................................................Passed
ruff.....................................................................Passed
fix end of files.........................................................Passed
[pre-commit-setup f00c0ce] testing
 1 file changed, 1 insertion(+), 1 deletion(-)
```

Now your file has been committed and you can push your changes.

At the beginning this might seem like a tedious process (having to add the file again after `black` and `ruff` have modified it) but it is actually very useful. It allows you to see what changes `black` and `ruff` have made to your files and make sure that they are correct before you commit them.

## Issue Tracker

If you encounter any bugs, issues, or have feature requests, please [create a new issue](https://github.com/AntonOsika/gpt-engineer/issues/new) on the project's GitHub repository. Provide a clear and descriptive title along with relevant details to help us address the problem or understand your request.

## Licensing

By contributing to GPT Engineer, you agree that your contributions will be licensed under the [LICENSE](../LICENSE) file of the project.

Thank you for your interest in contributing to GPT Engineer! We appreciate your support and look forward to your contributions.
