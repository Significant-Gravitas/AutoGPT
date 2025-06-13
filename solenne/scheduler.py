from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from . import ai
from . import tools


class Scheduler:
    def __init__(self) -> None:
        self._task: threading.Timer | None = None
        self._last_summary = 0.0

    def start(self) -> None:
        self._schedule()

    def _schedule(self) -> None:
        self._task = threading.Timer(3600, self._run)
        self._task.daemon = True
        self._task.start()

    def _run(self) -> None:
        snap = tools.sys_snapshot()
        msg = (
            f"System snapshot: CPU {snap.get('cpu_pct')}%, RAM {snap.get('ram_pct')}%, "
            f"Disk free {snap.get('disk_free_pct')}%, Uptime {snap.get('uptime')}"
        )
        for q in ai.async_messages.values():
            q.append(msg)
        now = time.time()
        if now - self._last_summary > 86400:
            path = Path("memory_emotional.jsonl")
            path.write_text(
                "Daily summary: system operational", encoding="utf-8", errors="ignore"
            )
            self._last_summary = now
        self._schedule()
