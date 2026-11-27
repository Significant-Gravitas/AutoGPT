from backend.app import run_processes
from backend.copilot.swap_service import SwapCredentialService
from backend.data.db_manager import DatabaseManager


def main():
    """
    Run the database service, and beside it the one the credential swap proxy
    calls: that one needs the same database and encryption key, and keeping it
    here keeps the proxy off every other method DatabaseManager exposes.
    """
    run_processes(SwapCredentialService().set_log_level("warning"), DatabaseManager())


if __name__ == "__main__":
    main()
