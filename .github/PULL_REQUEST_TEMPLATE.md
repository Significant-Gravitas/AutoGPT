### Background

<!-- Provide a concise overview of the rationale behind this change. Include relevant context, prior discussions, or links to related issues. Ensure that the change aligns with the project's overall direction. -->

### Changes

<!-- Describe the specific, focused change made in this pull request. Detail the modifications clearly and avoid any unrelated or "extra" changes. -->

### PR Quality Checklist

- [ ] I have run the following commands against my code to ensure it passes our linters:
  ```shell
  black . --exclude test.py
  isort .
  mypy .
  autoflake --remove-all-unused-imports --recursive --ignore-init-module-imports --ignore-pass-after-docstring --in-place agbenchmark
  ```
