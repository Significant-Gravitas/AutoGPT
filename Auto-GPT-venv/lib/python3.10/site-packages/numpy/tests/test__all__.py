
import collections
import numpy as np


def test_no_duplicates_in_np__all__():
    # Regression test for gh-10198.
    dups = {k: v for k, v in collections.Counter(np.__all__).items() if v > 1}
    assert len(dups) == 0
