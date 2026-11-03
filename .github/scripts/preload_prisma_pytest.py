import sys

import prisma

loaded = sorted(name for name in sys.modules if name == "backend" or name.startswith("backend."))
if loaded:
    raise RuntimeError(f"Prisma preloading imported measured application modules: {loaded}")

import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
