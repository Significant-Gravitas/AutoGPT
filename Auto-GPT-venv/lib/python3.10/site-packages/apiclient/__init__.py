"""Retain apiclient as an alias for googleapiclient."""

from googleapiclient import channel, discovery, errors, http, mimeparse, model

try:
    from googleapiclient import sample_tools
except ImportError:
    # Silently ignore, because the vast majority of consumers won't use it and
    # it has deep dependence on oauth2client, an optional dependency.
    sample_tools = None
from googleapiclient import schema

_SUBMODULES = {
    "channel": channel,
    "discovery": discovery,
    "errors": errors,
    "http": http,
    "mimeparse": mimeparse,
    "model": model,
    "sample_tools": sample_tools,
    "schema": schema,
}

import sys

for module_name, module in _SUBMODULES.items():
    sys.modules["apiclient.%s" % module_name] = module
