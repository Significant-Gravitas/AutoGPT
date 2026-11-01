"""Turns off the usage telemetry graphiti-core and mem0 send to their vendors.

Both libraries report usage to their own PostHog projects unless told not to.
mem0 reads its switch once, when it is first imported, so this has to run
before anything imports mem0; ``backend/__init__.py`` calls it, which puts it
ahead of every import of either library. A variable already in the process
environment keeps its value. One only in ``backend/.env`` does not: the
``load_dotenv()`` calls run after this and never overwrite a variable that is
already set.
"""

import os
from collections.abc import MutableMapping

VENDOR_TELEMETRY_OFF = {
    "GRAPHITI_TELEMETRY_ENABLED": "false",
    "MEM0_TELEMETRY": "false",
}


def disable_vendor_telemetry(environ: MutableMapping[str, str] = os.environ) -> None:
    for name, value in VENDOR_TELEMETRY_OFF.items():
        environ.setdefault(name, value)
