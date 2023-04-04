# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for commonly used utilities."""

import functools
import inspect
import logging
import urllib

logger = logging.getLogger(__name__)

POSITIONAL_WARNING = "WARNING"
POSITIONAL_EXCEPTION = "EXCEPTION"
POSITIONAL_IGNORE = "IGNORE"
POSITIONAL_SET = frozenset(
    [POSITIONAL_WARNING, POSITIONAL_EXCEPTION, POSITIONAL_IGNORE]
)

positional_parameters_enforcement = POSITIONAL_WARNING

_SYM_LINK_MESSAGE = "File: {0}: Is a symbolic link."
_IS_DIR_MESSAGE = "{0}: Is a directory"
_MISSING_FILE_MESSAGE = "Cannot access {0}: No such file or directory"


def positional(max_positional_args):
    """A decorator to declare that only the first N arguments may be positional.

    This decorator makes it easy to support Python 3 style keyword-only
    parameters. For example, in Python 3 it is possible to write::

        def fn(pos1, *, kwonly1=None, kwonly2=None):
            ...

    All named parameters after ``*`` must be a keyword::

        fn(10, 'kw1', 'kw2')  # Raises exception.
        fn(10, kwonly1='kw1')  # Ok.

    Example
    ^^^^^^^

    To define a function like above, do::

        @positional(1)
        def fn(pos1, kwonly1=None, kwonly2=None):
            ...

    If no default value is provided to a keyword argument, it becomes a
    required keyword argument::

        @positional(0)
        def fn(required_kw):
            ...

    This must be called with the keyword parameter::

        fn()  # Raises exception.
        fn(10)  # Raises exception.
        fn(required_kw=10)  # Ok.

    When defining instance or class methods always remember to account for
    ``self`` and ``cls``::

        class MyClass(object):

            @positional(2)
            def my_method(self, pos1, kwonly1=None):
                ...

            @classmethod
            @positional(2)
            def my_method(cls, pos1, kwonly1=None):
                ...

    The positional decorator behavior is controlled by
    ``_helpers.positional_parameters_enforcement``, which may be set to
    ``POSITIONAL_EXCEPTION``, ``POSITIONAL_WARNING`` or
    ``POSITIONAL_IGNORE`` to raise an exception, log a warning, or do
    nothing, respectively, if a declaration is violated.

    Args:
        max_positional_arguments: Maximum number of positional arguments. All
                                  parameters after this index must be
                                  keyword only.

    Returns:
        A decorator that prevents using arguments after max_positional_args
        from being used as positional parameters.

    Raises:
        TypeError: if a keyword-only argument is provided as a positional
                   parameter, but only if
                   _helpers.positional_parameters_enforcement is set to
                   POSITIONAL_EXCEPTION.
    """

    def positional_decorator(wrapped):
        @functools.wraps(wrapped)
        def positional_wrapper(*args, **kwargs):
            if len(args) > max_positional_args:
                plural_s = ""
                if max_positional_args != 1:
                    plural_s = "s"
                message = (
                    "{function}() takes at most {args_max} positional "
                    "argument{plural} ({args_given} given)".format(
                        function=wrapped.__name__,
                        args_max=max_positional_args,
                        args_given=len(args),
                        plural=plural_s,
                    )
                )
                if positional_parameters_enforcement == POSITIONAL_EXCEPTION:
                    raise TypeError(message)
                elif positional_parameters_enforcement == POSITIONAL_WARNING:
                    logger.warning(message)
            return wrapped(*args, **kwargs)

        return positional_wrapper

    if isinstance(max_positional_args, int):
        return positional_decorator
    else:
        args, _, _, defaults, _, _, _ = inspect.getfullargspec(max_positional_args)
        return positional(len(args) - len(defaults))(max_positional_args)


def parse_unique_urlencoded(content):
    """Parses unique key-value parameters from urlencoded content.

    Args:
        content: string, URL-encoded key-value pairs.

    Returns:
        dict, The key-value pairs from ``content``.

    Raises:
        ValueError: if one of the keys is repeated.
    """
    urlencoded_params = urllib.parse.parse_qs(content)
    params = {}
    for key, value in urlencoded_params.items():
        if len(value) != 1:
            msg = "URL-encoded content contains a repeated value:" "%s -> %s" % (
                key,
                ", ".join(value),
            )
            raise ValueError(msg)
        params[key] = value[0]
    return params


def update_query_params(uri, params):
    """Updates a URI with new query parameters.

    If a given key from ``params`` is repeated in the ``uri``, then
    the URI will be considered invalid and an error will occur.

    If the URI is valid, then each value from ``params`` will
    replace the corresponding value in the query parameters (if
    it exists).

    Args:
        uri: string, A valid URI, with potential existing query parameters.
        params: dict, A dictionary of query parameters.

    Returns:
        The same URI but with the new query parameters added.
    """
    parts = urllib.parse.urlparse(uri)
    query_params = parse_unique_urlencoded(parts.query)
    query_params.update(params)
    new_query = urllib.parse.urlencode(query_params)
    new_parts = parts._replace(query=new_query)
    return urllib.parse.urlunparse(new_parts)


def _add_query_parameter(url, name, value):
    """Adds a query parameter to a url.

    Replaces the current value if it already exists in the URL.

    Args:
        url: string, url to add the query parameter to.
        name: string, query parameter name.
        value: string, query parameter value.

    Returns:
        Updated query parameter. Does not update the url if value is None.
    """
    if value is None:
        return url
    else:
        return update_query_params(url, {name: value})
