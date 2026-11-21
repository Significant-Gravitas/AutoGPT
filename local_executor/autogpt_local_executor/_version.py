"""The package's version, one value for `--version`, HELLO and the audit log."""

from importlib.metadata import PackageNotFoundError, version

# pyproject.toml's version, for a source tree that isn't installed; a test
# (tests/test_packaging.py) checks that the two match.
SOURCE_TREE_VERSION = "0.0.1.dev0"


def distribution_version() -> str:
    """The installed distribution's version: pyproject.toml's, which the
    release workflow checks against the tag. A source tree that isn't
    installed has SOURCE_TREE_VERSION."""
    try:
        return version("autogpt-local-executor")
    except PackageNotFoundError:
        return SOURCE_TREE_VERSION


__version__ = distribution_version()
