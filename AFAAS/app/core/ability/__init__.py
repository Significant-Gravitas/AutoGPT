import warnings

from .. import tools

warnings.warn(
    "autogpt.core.ability is deprecated and will be removed in a future version. "
    "Use autogpt.core.tools instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Set up everything from tools to be accessible via ability
from ..tools import *
