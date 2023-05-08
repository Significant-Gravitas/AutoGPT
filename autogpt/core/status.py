"""Temp Status Enum to keep track of our progress"""
import enum

class Status(enum.Enum):
    """Enum for the status of a project."""
    TODO = 0
    IN_PROGRESS = 1
    INTERFACE_DONE = 2
    BASIC_DONE = 3
    TESTING = 4
    RELEASE_READY = 5