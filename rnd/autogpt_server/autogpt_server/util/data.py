import os
import pathlib
import sys


def get_secrets_path() -> pathlib.Path:
    return get_data_path() / "secrets"


def get_config_path() -> pathlib.Path:
    return get_data_path()


def get_frontend_path() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        # The application is frozen
        datadir = pathlib.Path(os.path.dirname(sys.executable)) / "frontend"
    else:
        # The application is not frozen
        # Change this bit to match where you store your data files:
        filedir = os.path.dirname(__file__)
        datadir = pathlib.Path(filedir).parent.parent.parent / "frontend"
    return pathlib.Path(datadir)


def get_prisma_path() -> pathlib.Path:
    return get_data_path() / "prisma"


def get_prisma_exe_path() -> pathlib.Path:
    if sys.platform == "win32":
        return (
            get_prisma_path() / "node_modules" / "prisma" / "query-engine-windows.exe"
        )
    else:
        raise NotImplementedError(
            "Freezing AutoGPT Server is only supported on Windows at the moment."
        )


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
