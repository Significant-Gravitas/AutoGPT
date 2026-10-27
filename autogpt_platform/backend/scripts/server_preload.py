"""Share imported test-service code through the existing forkserver."""

import gc
import importlib

importlib.import_module("pytest")
importlib.import_module("backend.util.test")
importlib.import_module("backend.blocks").load_all_blocks()

# Keep cyclic GC from dirtying inherited objects and defeating copy-on-write.
gc.collect()
gc.freeze()
