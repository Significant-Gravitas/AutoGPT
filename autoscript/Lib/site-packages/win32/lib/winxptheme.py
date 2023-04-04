"""A useful wrapper around the "_winxptheme" module.

Originally used when we couldn't be sure Windows XP apis were going to
be available. In 2022, it's safe to assume they are, so this is just a wrapper
around _winxptheme.
"""
from _winxptheme import *
