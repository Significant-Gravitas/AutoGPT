"""
Enables multiple commonly used features.

Method resolution order:

- `tqdm.autonotebook` without import warnings
- `tqdm.asyncio`
- `tqdm.std` base class

Usage:
>>> from tqdm.auto import trange, tqdm
>>> for i in trange(10):
...     ...
"""
import warnings

from .std import TqdmExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)
    from .autonotebook import tqdm as notebook_tqdm

from .asyncio import tqdm as asyncio_tqdm
from .std import tqdm as std_tqdm

if notebook_tqdm != std_tqdm:
    class tqdm(notebook_tqdm, asyncio_tqdm):  # pylint: disable=inconsistent-mro
        pass
else:
    tqdm = asyncio_tqdm


def trange(*args, **kwargs):
    """
    A shortcut for `tqdm.auto.tqdm(range(*args), **kwargs)`.
    """
    return tqdm(range(*args), **kwargs)


__all__ = ["tqdm", "trange"]
