"""Which install of the app this runtime belongs to.

The app can be installed more than once on a machine: the normal app, and
variants of it that share nothing with it (src/identity.js). The shell names
its install in the environment. The runtime uses the name for the few things
that live outside the data directory and would otherwise be shared: a folder
under the cache directory (rabbitmq.py) and the browser's cookies (cookies.py).
"""

from __future__ import annotations

import os
import re

VARIABLE = "AUTOGPT_DESKTOP_INSTALL_NAME"
NORMAL = "AutoGPT"
# `AutoGPT`, or `AutoGPT-<slug>` for a variant: one path component.
NAME = re.compile(r"AutoGPT(-[a-z0-9-]+)?")


def name() -> str:
    """The name the shell gave, or the normal app's when there is none or it
    is not a name: it becomes part of a path and of cookie names."""
    given = os.environ.get(VARIABLE, NORMAL)
    return given if NAME.fullmatch(given) else NORMAL
