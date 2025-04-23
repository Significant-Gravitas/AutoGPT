import os
import pathlib
import sys


def get_frontend_path() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        # The application is frozen
        datadir = pathlib.Path(os.path.dirname(sys.executable)) / "example_files"
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        filedir = os.path.dirname(__file__)
        datadir = pathlib.Path(filedir).parent.parent.parent / "example_files"
    return pathlib.Path(datadir)


def get_data_path() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        # The application is frozen
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        filedir = os.path.dirname(__file__)
        datadir = pathlib.Path(filedir).parent.parent
    return pathlib.Path(datadir)
