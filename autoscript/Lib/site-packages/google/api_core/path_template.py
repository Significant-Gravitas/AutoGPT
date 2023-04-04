# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Expand and validate URL path templates.

This module provides the :func:`expand` and :func:`validate` functions for
interacting with Google-style URL `path templates`_ which are commonly used
in Google APIs for `resource names`_.

.. _path templates: https://github.com/googleapis/googleapis/blob
    /57e2d376ac7ef48681554204a3ba78a414f2c533/google/api/http.proto#L212
.. _resource names: https://cloud.google.com/apis/design/resource_names
"""

from __future__ import unicode_literals

from collections import deque
import copy
import functools
import re

# Regular expression for extracting variable parts from a path template.
# The variables can be expressed as:
#
# - "*": a single-segment positional variable, for example: "books/*"
# - "**": a multi-segment positional variable, for example: "shelf/**/book/*"
# - "{name}": a single-segment wildcard named variable, for example
#   "books/{name}"
# - "{name=*}: same as above.
# - "{name=**}": a multi-segment wildcard named variable, for example
#   "shelf/{name=**}"
# - "{name=/path/*/**}": a multi-segment named variable with a sub-template.
_VARIABLE_RE = re.compile(
    r"""
    (  # Capture the entire variable expression
        (?P<positional>\*\*?)  # Match & capture * and ** positional variables.
        |
        # Match & capture named variables {name}
        {
            (?P<name>[^/]+?)
            # Optionally match and capture the named variable's template.
            (?:=(?P<template>.+?))?
        }
    )
    """,
    re.VERBOSE,
)

# Segment expressions used for validating paths against a template.
_SINGLE_SEGMENT_PATTERN = r"([^/]+)"
_MULTI_SEGMENT_PATTERN = r"(.+)"


def _expand_variable_match(positional_vars, named_vars, match):
    """Expand a matched variable with its value.

    Args:
        positional_vars (list): A list of positional variables. This list will
            be modified.
        named_vars (dict): A dictionary of named variables.
        match (re.Match): A regular expression match.

    Returns:
        str: The expanded variable to replace the match.

    Raises:
        ValueError: If a positional or named variable is required by the
            template but not specified or if an unexpected template expression
            is encountered.
    """
    positional = match.group("positional")
    name = match.group("name")
    if name is not None:
        try:
            return str(named_vars[name])
        except KeyError:
            raise ValueError(
                "Named variable '{}' not specified and needed by template "
                "`{}` at position {}".format(name, match.string, match.start())
            )
    elif positional is not None:
        try:
            return str(positional_vars.pop(0))
        except IndexError:
            raise ValueError(
                "Positional variable not specified and needed by template "
                "`{}` at position {}".format(match.string, match.start())
            )
    else:
        raise ValueError("Unknown template expression {}".format(match.group(0)))


def expand(tmpl, *args, **kwargs):
    """Expand a path template with the given variables.

    .. code-block:: python

        >>> expand('users/*/messages/*', 'me', '123')
        users/me/messages/123
        >>> expand('/v1/{name=shelves/*/books/*}', name='shelves/1/books/3')
        /v1/shelves/1/books/3

    Args:
        tmpl (str): The path template.
        args: The positional variables for the path.
        kwargs: The named variables for the path.

    Returns:
        str: The expanded path

    Raises:
        ValueError: If a positional or named variable is required by the
            template but not specified or if an unexpected template expression
            is encountered.
    """
    replacer = functools.partial(_expand_variable_match, list(args), kwargs)
    return _VARIABLE_RE.sub(replacer, tmpl)


def _replace_variable_with_pattern(match):
    """Replace a variable match with a pattern that can be used to validate it.

    Args:
        match (re.Match): A regular expression match

    Returns:
        str: A regular expression pattern that can be used to validate the
            variable in an expanded path.

    Raises:
        ValueError: If an unexpected template expression is encountered.
    """
    positional = match.group("positional")
    name = match.group("name")
    template = match.group("template")
    if name is not None:
        if not template:
            return _SINGLE_SEGMENT_PATTERN.format(name)
        elif template == "**":
            return _MULTI_SEGMENT_PATTERN.format(name)
        else:
            return _generate_pattern_for_template(template)
    elif positional == "*":
        return _SINGLE_SEGMENT_PATTERN
    elif positional == "**":
        return _MULTI_SEGMENT_PATTERN
    else:
        raise ValueError("Unknown template expression {}".format(match.group(0)))


def _generate_pattern_for_template(tmpl):
    """Generate a pattern that can validate a path template.

    Args:
        tmpl (str): The path template

    Returns:
        str: A regular expression pattern that can be used to validate an
            expanded path template.
    """
    return _VARIABLE_RE.sub(_replace_variable_with_pattern, tmpl)


def get_field(request, field):
    """Get the value of a field from a given dictionary.

    Args:
        request (dict | Message): A dictionary or a Message object.
        field (str): The key to the request in dot notation.

    Returns:
        The value of the field.
    """
    parts = field.split(".")
    value = request

    for part in parts:
        if not isinstance(value, dict):
            value = getattr(value, part, None)
        else:
            value = value.get(part)
    if isinstance(value, dict):
        return
    return value


def delete_field(request, field):
    """Delete the value of a field from a given dictionary.

    Args:
        request (dict | Message): A dictionary object or a Message.
        field (str): The key to the request in dot notation.
    """
    parts = deque(field.split("."))
    while len(parts) > 1:
        part = parts.popleft()
        if not isinstance(request, dict):
            if hasattr(request, part):
                request = getattr(request, part, None)
            else:
                return
        else:
            request = request.get(part)
    part = parts.popleft()
    if not isinstance(request, dict):
        if hasattr(request, part):
            request.ClearField(part)
        else:
            return
    else:
        request.pop(part, None)


def validate(tmpl, path):
    """Validate a path against the path template.

    .. code-block:: python

        >>> validate('users/*/messages/*', 'users/me/messages/123')
        True
        >>> validate('users/*/messages/*', 'users/me/drafts/123')
        False
        >>> validate('/v1/{name=shelves/*/books/*}', /v1/shelves/1/books/3)
        True
        >>> validate('/v1/{name=shelves/*/books/*}', /v1/shelves/1/tapes/3)
        False

    Args:
        tmpl (str): The path template.
        path (str): The expanded path.

    Returns:
        bool: True if the path matches.
    """
    pattern = _generate_pattern_for_template(tmpl) + "$"
    return True if re.match(pattern, path) is not None else False


def transcode(http_options, message=None, **request_kwargs):
    """Transcodes a grpc request pattern into a proper HTTP request following the rules outlined here,
    https://github.com/googleapis/googleapis/blob/master/google/api/http.proto#L44-L312

     Args:
         http_options (list(dict)): A list of dicts which consist of these keys,
             'method'    (str): The http method
             'uri'       (str): The path template
             'body'      (str): The body field name (optional)
             (This is a simplified representation of the proto option `google.api.http`)

         message (Message) : A request object (optional)
         request_kwargs (dict) : A dict representing the request object

     Returns:
         dict: The transcoded request with these keys,
             'method'        (str)   : The http method
             'uri'           (str)   : The expanded uri
             'body'          (dict | Message)  : A dict or a Message representing the body (optional)
             'query_params'  (dict | Message)  : A dict or Message mapping query parameter variables and values

     Raises:
         ValueError: If the request does not match the given template.
    """
    transcoded_value = message or request_kwargs
    bindings = []
    for http_option in http_options:
        request = {}

        # Assign path
        uri_template = http_option["uri"]
        fields = [
            (m.group("name"), m.group("template"))
            for m in _VARIABLE_RE.finditer(uri_template)
        ]
        bindings.append((uri_template, fields))

        path_args = {field: get_field(transcoded_value, field) for field, _ in fields}
        request["uri"] = expand(uri_template, **path_args)

        if not validate(uri_template, request["uri"]) or not all(path_args.values()):
            continue

        # Remove fields used in uri path from request
        leftovers = copy.deepcopy(transcoded_value)
        for path_field, _ in fields:
            delete_field(leftovers, path_field)

        # Assign body and query params
        body = http_option.get("body")

        if body:
            if body == "*":
                request["body"] = leftovers
                if message:
                    request["query_params"] = message.__class__()
                else:
                    request["query_params"] = {}
            else:
                try:
                    if message:
                        request["body"] = getattr(leftovers, body)
                        delete_field(leftovers, body)
                    else:
                        request["body"] = leftovers.pop(body)
                except (KeyError, AttributeError):
                    continue
                request["query_params"] = leftovers
        else:
            request["query_params"] = leftovers
        request["method"] = http_option["method"]
        return request

    bindings_description = [
        '\n\tURI: "{}"'
        "\n\tRequired request fields:\n\t\t{}".format(
            uri,
            "\n\t\t".join(
                [
                    'field: "{}", pattern: "{}"'.format(n, p if p else "*")
                    for n, p in fields
                ]
            ),
        )
        for uri, fields in bindings
    ]

    raise ValueError(
        "Invalid request."
        "\nSome of the fields of the request message are either not initialized or "
        "initialized with an invalid value."
        "\nPlease make sure your request matches at least one accepted HTTP binding."
        "\nTo match a binding the request message must have all the required fields "
        "initialized with values matching their patterns as listed below:{}".format(
            "\n".join(bindings_description)
        )
    )
