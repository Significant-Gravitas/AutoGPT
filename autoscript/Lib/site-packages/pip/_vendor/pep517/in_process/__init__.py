"""This is a subpackage because the directory is on sys.path for _in_process.py

The subpackage should stay as empty as possible to avoid shadowing modules that
the backend might import.
"""
from contextlib import contextmanager
from os.path import abspath, dirname
from os.path import join as pjoin

try:
    import importlib.resources as resources
    try:
        resources.files
    except AttributeError:
        # Python 3.8 compatibility
        def _in_proc_script_path():
            return resources.path(__package__, '_in_process.py')
    else:
        def _in_proc_script_path():
            return resources.as_file(
                resources.files(__package__).joinpath('_in_process.py'))
except ImportError:
    # Python 3.6 compatibility
    @contextmanager
    def _in_proc_script_path():
        yield pjoin(dirname(abspath(__file__)), '_in_process.py')
