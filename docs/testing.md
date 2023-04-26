# Running tests

To run all tests, use the following command:

``` shell
pytest
```

If `pytest` is not found:
``` shell
python -m pytest
```

### Running specific test suites

- To run without integration tests:

        :::shell
        pytest --without-integration

- To run without *slow* integration tests:

        :::shell
        pytest --without-slow-integration

- To run tests and see coverage:

        :::shell
        pytest --cov=autogpt --without-integration --without-slow-integration

## Runing the linter

This project uses [flake8](https://flake8.pycqa.org/en/latest/) for linting.
We currently use the following rules: `E303,W293,W291,W292,E305,E231,E302`.
See the [flake8 rules](https://www.flake8rules.com/) for more information.

To run the linter:

``` shell
flake8 .
```

Or:
``` shell
python -m flake8 .
```
