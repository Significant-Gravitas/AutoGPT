# Contributing to the Docs

We welcome contributions to our documentation! Our docs are hosted on GitBook and synced with GitHub.

## How It Works

- Documentation lives in the `docs/` directory on the `gitbook` branch
- GitBook automatically syncs changes from GitHub
- You can edit docs directly on GitHub or locally

## Editing Docs Locally

1. Clone the repository and switch to the gitbook branch:

    ```shell
    git clone https://github.com/Significant-Gravitas/AutoGPT.git
    cd AutoGPT
    git checkout gitbook
    ```

2. Make your changes to markdown files in `docs/`

3. Preview changes:
   - Push to a branch and create a PR - GitBook will generate a preview
   - Or use any markdown preview tool locally

## Adding a New Page

1. Create a new markdown file in the appropriate `docs/` subdirectory
2. Add the new page to the relevant `SUMMARY.md` file to include it in the navigation
3. Submit a pull request to the `gitbook` branch

## Submitting a Pull Request

When you're ready to submit your changes, create a pull request targeting the `gitbook` branch. We will review your changes and merge them if appropriate.
