"""Split upstream's database entry point into independently owned hosts."""


def database_main():
    from backend.app import run_processes
    from backend.data.db_manager import DatabaseManager

    run_processes(DatabaseManager())


def swap_main():
    from backend.app import run_processes
    from backend.copilot.swap_service import SwapCredentialService

    run_processes(SwapCredentialService().set_log_level("warning"))
