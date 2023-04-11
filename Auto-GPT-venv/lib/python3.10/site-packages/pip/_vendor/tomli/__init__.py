"""A lil' TOML parser."""

__all__ = ("loads", "load", "TOMLDecodeError")
__version__ = "1.0.3"  # DO NOT EDIT THIS LINE MANUALLY. LET bump2version UTILITY DO IT

from pip._vendor.tomli._parser import TOMLDecodeError, load, loads
