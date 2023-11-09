"""

uritemplate.template
====================

This module contains the essential inner workings of uritemplate.

What treasures await you:

- URITemplate class

You see a treasure chest of knowledge in front of you.
What do you do?
>

"""
import re
import typing as t

from uritemplate import orderedset
from uritemplate import variable

template_re = re.compile("{([^}]+)}")


def _merge(
    var_dict: t.Optional[variable.VariableValueDict],
    overrides: variable.VariableValueDict,
) -> variable.VariableValueDict:
    if var_dict:
        opts = var_dict.copy()
        opts.update(overrides)
        return opts
    return overrides


class URITemplate:

    """This parses the template and will be used to expand it.

    This is the most important object as the center of the API.

    Example::

        from uritemplate import URITemplate
        import requests


        t = URITemplate(
            'https://api.github.com/users/sigmavirus24/gists{/gist_id}'
        )
        uri = t.expand(gist_id=123456)
        resp = requests.get(uri)
        for gist in resp.json():
            print(gist['html_url'])

    Please note::

        str(t)
        # 'https://api.github.com/users/sigmavirus24/gists{/gistid}'
        repr(t)  # is equivalent to
        # URITemplate(str(t))
        # Where str(t) is interpreted as the URI string.

    Also, ``URITemplates`` are hashable so they can be used as keys in
    dictionaries.

    """

    def __init__(self, uri: str):
        #: The original URI to be parsed.
        self.uri: str = uri
        #: A list of the variables in the URI. They are stored as
        #: :class:`~uritemplate.variable.URIVariable`\ s
        self.variables: t.List[variable.URIVariable] = [
            variable.URIVariable(m.groups()[0])
            for m in template_re.finditer(self.uri)
        ]
        #: A set of variable names in the URI.
        self.variable_names = orderedset.OrderedSet()
        for var in self.variables:
            for name in var.variable_names:
                self.variable_names.add(name)

    def __repr__(self) -> str:
        return 'URITemplate("%s")' % self

    def __str__(self) -> str:
        return self.uri

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, URITemplate):
            return NotImplemented
        return self.uri == other.uri

    def __hash__(self) -> int:
        return hash(self.uri)

    def _expand(
        self, var_dict: variable.VariableValueDict, replace: bool
    ) -> str:
        if not self.variables:
            return self.uri

        expansion = var_dict
        expanded: t.Dict[str, str] = {}
        for v in self.variables:
            expanded.update(v.expand(expansion))

        def replace_all(match: "re.Match[str]") -> str:
            return expanded.get(match.groups()[0], "")

        def replace_partial(match: "re.Match[str]") -> str:
            match_group = match.groups()[0]
            var = "{%s}" % match_group
            return expanded.get(match_group) or var

        replace_func = replace_partial if replace else replace_all

        return template_re.sub(replace_func, self.uri)

    def expand(
        self,
        var_dict: t.Optional[variable.VariableValueDict] = None,
        **kwargs: variable.VariableValue,
    ) -> str:
        """Expand the template with the given parameters.

        :param dict var_dict: Optional dictionary with variables and values
        :param kwargs: Alternative way to pass arguments
        :returns: str

        Example::

            t = URITemplate('https://api.github.com{/end}')
            t.expand({'end': 'users'})
            t.expand(end='gists')

        .. note:: Passing values by both parts, may override values in
                  ``var_dict``. For example::

                      expand('https://{var}', {'var': 'val1'}, var='val2')

                  ``val2`` will be used instead of ``val1``.

        """
        return self._expand(_merge(var_dict, kwargs), False)

    def partial(
        self,
        var_dict: t.Optional[variable.VariableValueDict] = None,
        **kwargs: variable.VariableValue,
    ) -> "URITemplate":
        """Partially expand the template with the given parameters.

        If all of the parameters for the template are not given, return a
        partially expanded template.

        :param dict var_dict: Optional dictionary with variables and values
        :param kwargs: Alternative way to pass arguments
        :returns: :class:`URITemplate`

        Example::

            t = URITemplate('https://api.github.com{/end}')
            t.partial()  # => URITemplate('https://api.github.com{/end}')

        """
        return URITemplate(self._expand(_merge(var_dict, kwargs), True))
