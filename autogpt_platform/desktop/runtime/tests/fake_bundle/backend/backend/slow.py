"""An entry module that takes as long to import as the real ones."""

import time

from backend.app import run_processes
from backend.services import Service, mark

mark("slow.importing")
time.sleep(20)


def main() -> None:
    run_processes(Service("slow"))
