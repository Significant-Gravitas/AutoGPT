try:
    from ._version import __version__
except ImportError:
    try:
        # importlib.metadata available in Python 3.8+, the fallback (0.0.0)
        # is fine because release builds use _version (above) rather than
        # this code path, so it only impacts developing w/ 3.7
        from importlib.metadata import version, PackageNotFoundError
        try:
            __version__ = version('docker')
        except PackageNotFoundError:
            __version__ = '0.0.0'
    except ImportError:
        __version__ = '0.0.0'
