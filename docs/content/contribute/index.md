# Contributing to the Docs

We welcome contributions to our documentation! Our docs are hosted on GitBook and published from GitHub.

## How It Works

- Documentation lives in the `docs/` directory and changes like any other code: through a pull request to `dev`
- The published docs match `master`. When a release reaches `master`, its `docs/` directory is copied to the `gitbook` branch, and GitBook publishes that
- The `gitbook` branch belongs to that copy. Don't open pull requests against it, and don't edit pages in GitBook: the next copy overwrites anything that isn't in `master`

## Editing Docs Locally

1. Clone the repository and create a branch from `dev`:

    ```shell
    git clone https://github.com/Significant-Gravitas/AutoGPT.git
    cd AutoGPT
    git checkout -b my-docs-change origin/dev
    ```

2. Make your changes to markdown files in `docs/`

3. Preview changes with any markdown preview tool

## Adding a New Page

1. Create a new markdown file in the appropriate `docs/` subdirectory
2. Add the new page to the relevant `SUMMARY.md` file to include it in the navigation
3. Submit a pull request to the `dev` branch

## Submitting a Pull Request

When you're ready to submit your changes, create a pull request targeting the `dev` branch. We will review your changes and merge them if appropriate. They appear on the published docs with the next release.
