from numpy.testing import (
    assert_raises,
    assert_warns,
    assert_,
    assert_equal,
    IS_WASM,
)
from numpy.compat import pickle

import pytest
import sys
import subprocess
import textwrap
from importlib import reload


def test_numpy_reloading():
    # gh-7844. Also check that relevant globals retain their identity.
    import numpy as np
    import numpy._globals

    _NoValue = np._NoValue
    VisibleDeprecationWarning = np.VisibleDeprecationWarning
    ModuleDeprecationWarning = np.ModuleDeprecationWarning

    with assert_warns(UserWarning):
        reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is np.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is np.VisibleDeprecationWarning)

    assert_raises(RuntimeError, reload, numpy._globals)
    with assert_warns(UserWarning):
        reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is np.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is np.VisibleDeprecationWarning)

def test_novalue():
    import numpy as np
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        assert_equal(repr(np._NoValue), '<no value>')
        assert_(pickle.loads(pickle.dumps(np._NoValue,
                                          protocol=proto)) is np._NoValue)


@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
def test_full_reimport():
    """At the time of writing this, it is *not* truly supported, but
    apparently enough users rely on it, for it to be an annoying change
    when it started failing previously.
    """
    # Test within a new process, to ensure that we do not mess with the
    # global state during the test run (could lead to cryptic test failures).
    # This is generally unsafe, especially, since we also reload the C-modules.
    code = textwrap.dedent(r"""
        import sys
        from pytest import warns
        import numpy as np

        for k in list(sys.modules.keys()):
            if "numpy" in k:
                del sys.modules[k]

        with warns(UserWarning):
            import numpy as np
        """)
    p = subprocess.run([sys.executable, '-c', code], capture_output=True)
    if p.returncode:
        raise AssertionError(
            f"Non-zero return code: {p.returncode!r}\n\n{p.stderr.decode()}"
        )
