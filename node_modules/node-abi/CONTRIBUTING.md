# Contributing to `node-abi`

:+1::tada: First off, thanks for taking the time to contribute to `node-abi`! :tada::+1:

## Commit Message Guidelines

This module uses [`semantic-release`](https://github.com/semantic-release/semantic-release) to automatically release new versions via [Continuous Auth](https://continuousauth.dev/).
Therefor we have very precise rules over how our git commit messages can be formatted.

Each commit message consists of a **header**, a **body** and a **footer**.  The header has a special
format that includes a **type**, a **scope** and a **subject** ([full explanation](https://github.com/stevemao/conventional-changelog-angular/blob/master/convention.md)):

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

### Type

Must be one of the following:

- **feat**: A new feature. **Will trigger a new release**
- **fix**: A bug fix or a addition to one of the target arrays. **Will trigger a new release**
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools and libraries such as documentation generation


### Patch Release

```
fix(electron): Support Electron 1.8.0
```

### ~~Minor~~ Feature Release

```
feat: add .getTarget(abi, runtime)
```

### ~~Major~~ Breaking Release

```
feat: Add amazing new feature

BREAKING CHANGE: This removes support for Node 0.10 and 0.12.
```

## Pull request guidelines

Here are some things to keep in mind as you file pull requests to fix bugs, add new features, etc.:

- CircleCI is used to make sure that the project builds packages as expected on the supported platforms, using supported Node.js versions.
- Unless it's impractical, please write tests for your changes. This will help us so that we can spot regressions much easier.
- Squashing commits during discussion of the pull request is almost always unnecessary, and makes it more difficult for both the submitters and reviewers to understand what changed in between comments. However, rebasing is encouraged when practical, particularly when there's a merge conflict.
- If you are continuing the work of another person's PR and need to rebase/squash, please retain the attribution of the original author(s) and continue the work in subsequent commits.
