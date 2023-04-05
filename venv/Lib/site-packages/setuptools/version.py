from ._importlib import metadata

try:
    __version__ = metadata.version('setuptools')
except Exception:
    __version__ = 'unknown'
