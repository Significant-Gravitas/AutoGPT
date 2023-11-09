"""
requests.compat
~~~~~~~~~~~~~~~

This module previously handled import compatibility issues
between Python 2 and Python 3. It remains for backwards
compatibility until the next major version.
"""

try:
    import chardet
except ImportError:
    import charset_normalizer as chardet

import sys

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = _ver[0] == 2

#: Python 3.x?
is_py3 = _ver[0] == 3

# json/simplejson module import resolution
has_simplejson = False
try:
    import simplejson as json

    has_simplejson = True
except ImportError:
    import json

if has_simplejson:
    from simplejson import JSONDecodeError
else:
    from json import JSONDecodeError

# Keep OrderedDict for backwards compatibility.
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from http import cookiejar as cookielib
from http.cookies import Morsel
from io import StringIO

# --------------
# Legacy Imports
# --------------
from urllib.parse import (
    quote,
    quote_plus,
    unquote,
    unquote_plus,
    urldefrag,
    urlencode,
    urljoin,
    urlparse,
    urlsplit,
    urlunparse,
)
from urllib.request import (
    getproxies,
    getproxies_environment,
    parse_http_list,
    proxy_bypass,
    proxy_bypass_environment,
)

builtin_str = str
str = str
bytes = bytes
basestring = (str, bytes)
numeric_types = (int, float)
integer_types = (int,)
