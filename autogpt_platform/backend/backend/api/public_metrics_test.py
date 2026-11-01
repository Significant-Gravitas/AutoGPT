"""No public app routes /metrics: the servers serve them on ports of their own.

The route would be added when a module is imported, so the probe imports the
apps in a fresh process with metrics on.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

_PUBLIC_ROUTES_PROBE = """
from starlette.routing import Mount

from backend.api import rest_api, ws_api


def paths(routes, prefix=""):
    for route in routes:
        yield prefix + route.path
        if isinstance(route, Mount):
            yield from paths(route.routes, prefix + route.path)


print(sorted(set(paths(rest_api.app.routes)) | set(paths(ws_api.app.routes))))
"""


def test_no_public_app_routes_metrics() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", _PUBLIC_ROUTES_PROBE],
        cwd=Path(__file__).parents[2],
        env={**os.environ, "ENABLE_METRICS": "true"},
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert probe.returncode == 0, probe.stderr
    paths = ast.literal_eval(probe.stdout.strip().splitlines()[-1])
    # The probe saw the real apps, the mounted external API's routes included.
    assert {"/health", "/ws"} <= set(paths), paths
    assert any(path.startswith("/external-api/") for path in paths), paths
    assert [path for path in paths if path.endswith("/metrics")] == []
