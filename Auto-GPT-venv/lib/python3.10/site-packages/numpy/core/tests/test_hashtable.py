import pytest

import random
from numpy.core._multiarray_tests import identityhash_tester


@pytest.mark.parametrize("key_length", [1, 3, 6])
@pytest.mark.parametrize("length", [1, 16, 2000])
def test_identity_hashtable(key_length, length):
    # use a 30 object pool for everything (duplicates will happen)
    pool = [object() for i in range(20)]
    keys_vals = []
    for i in range(length):
        keys = tuple(random.choices(pool, k=key_length))
        keys_vals.append((keys, random.choice(pool)))

    dictionary = dict(keys_vals)

    # add a random item at the end:
    keys_vals.append(random.choice(keys_vals))
    # the expected one could be different with duplicates:
    expected = dictionary[keys_vals[-1][0]]

    res = identityhash_tester(key_length, keys_vals, replace=True)
    assert res is expected

    # check that ensuring one duplicate definitely raises:
    keys_vals.insert(0, keys_vals[-2])
    with pytest.raises(RuntimeError):
        identityhash_tester(key_length, keys_vals)
