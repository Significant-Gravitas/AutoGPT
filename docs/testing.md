## Run tests

To run all tests, run the following command:

```
pytest 
```

To run just without integration tests:

```
pytest --without-integration
```

To run just without slow integration tests:

```
pytest --without-slow-integration
```

To run tests and see coverage, run the following command:

```
pytest --cov=autogpt --without-integration --without-slow-integration
```

## Run linter

This project uses [flake8](https://flake8.pycqa.org/en/latest/) for linting. We currently use the following rules: `E303,W293,W291,W292,E305,E231,E302`. See the [flake8 rules](https://www.flake8rules.com/) for more information.

To run the linter, run the following command:

```
flake8 autogpt/ tests/

# Or, if you want to run flake8 with the same configuration as the CI:

flake8 autogpt/ tests/ --select E303,W293,W291,W292,E305,E231,E302
```