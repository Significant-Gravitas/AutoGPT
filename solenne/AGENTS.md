# Contributor Guidelines

When modifying this repository, run the following checks before committing:

```
pre-commit run --files <changed files>  # if pre-commit is installed
pytest
mypy --config-file mypy.ini
```

If `pre-commit` isn't available, you may skip it but still run `pytest` and `mypy`.
