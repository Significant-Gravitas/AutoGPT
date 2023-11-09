import functools

from .. import errors
from . import utils


def check_resource(resource_name):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(self, resource_id=None, *args, **kwargs):
            if resource_id is None and kwargs.get(resource_name):
                resource_id = kwargs.pop(resource_name)
            if isinstance(resource_id, dict):
                resource_id = resource_id.get('Id', resource_id.get('ID'))
            if not resource_id:
                raise errors.NullResource(
                    'Resource ID was not provided'
                )
            return f(self, resource_id, *args, **kwargs)
        return wrapped
    return decorator


def minimum_version(version):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if utils.version_lt(self._version, version):
                raise errors.InvalidVersion(
                    '{} is not available for version < {}'.format(
                        f.__name__, version
                    )
                )
            return f(self, *args, **kwargs)
        return wrapper
    return decorator


def update_headers(f):
    def inner(self, *args, **kwargs):
        if 'HttpHeaders' in self._general_configs:
            if not kwargs.get('headers'):
                kwargs['headers'] = self._general_configs['HttpHeaders']
            else:
                kwargs['headers'].update(self._general_configs['HttpHeaders'])
        return f(self, *args, **kwargs)
    return inner
