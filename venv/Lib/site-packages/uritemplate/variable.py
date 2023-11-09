"""

uritemplate.variable
====================

This module contains the URIVariable class which powers the URITemplate class.

What treasures await you:

- URIVariable class

You see a hammer in front of you.
What do you do?
>

"""
import collections.abc
import typing as t
import urllib.parse

ScalarVariableValue = t.Union[int, float, complex, str]
VariableValue = t.Union[
    t.Sequence[ScalarVariableValue],
    t.Mapping[str, ScalarVariableValue],
    t.Tuple[str, ScalarVariableValue],
    ScalarVariableValue,
]
VariableValueDict = t.Dict[str, VariableValue]


class URIVariable:

    """This object validates everything inside the URITemplate object.

    It validates template expansions and will truncate length as decided by
    the template.

    Please note that just like the :class:`URITemplate <URITemplate>`, this
    object's ``__str__`` and ``__repr__`` methods do not return the same
    information. Calling ``str(var)`` will return the original variable.

    This object does the majority of the heavy lifting. The ``URITemplate``
    object finds the variables in the URI and then creates ``URIVariable``
    objects.  Expansions of the URI are handled by each ``URIVariable``
    object. ``URIVariable.expand()`` returns a dictionary of the original
    variable and the expanded value. Check that method's documentation for
    more information.

    """

    operators = ("+", "#", ".", "/", ";", "?", "&", "|", "!", "@")
    reserved = ":/?#[]@!$&'()*+,;="

    def __init__(self, var: str):
        #: The original string that comes through with the variable
        self.original: str = var
        #: The operator for the variable
        self.operator: str = ""
        #: List of safe characters when quoting the string
        self.safe: str = ""
        #: List of variables in this variable
        self.variables: t.List[
            t.Tuple[str, t.MutableMapping[str, t.Any]]
        ] = []
        #: List of variable names
        self.variable_names: t.List[str] = []
        #: List of defaults passed in
        self.defaults: t.MutableMapping[str, ScalarVariableValue] = {}
        # Parse the variable itself.
        self.parse()
        self.post_parse()

    def __repr__(self) -> str:
        return "URIVariable(%s)" % self

    def __str__(self) -> str:
        return self.original

    def parse(self) -> None:
        """Parse the variable.

        This finds the:
            - operator,
            - set of safe characters,
            - variables, and
            - defaults.

        """
        var_list_str = self.original
        if self.original[0] in URIVariable.operators:
            self.operator = self.original[0]
            var_list_str = self.original[1:]

        if self.operator in URIVariable.operators[:2]:
            self.safe = URIVariable.reserved

        var_list = var_list_str.split(",")

        for var in var_list:
            default_val = None
            name = var
            if "=" in var:
                name, default_val = tuple(var.split("=", 1))

            explode = False
            if name.endswith("*"):
                explode = True
                name = name[:-1]

            prefix: t.Optional[int] = None
            if ":" in name:
                name, prefix_str = tuple(name.split(":", 1))
                prefix = int(prefix_str)

            if default_val:
                self.defaults[name] = default_val

            self.variables.append(
                (name, {"explode": explode, "prefix": prefix})
            )

        self.variable_names = [varname for (varname, _) in self.variables]

    def post_parse(self) -> None:
        """Set ``start``, ``join_str`` and ``safe`` attributes.

        After parsing the variable, we need to set up these attributes and it
        only makes sense to do it in a more easily testable way.
        """
        self.safe = ""
        self.start = self.join_str = self.operator
        if self.operator == "+":
            self.start = ""
        if self.operator in ("+", "#", ""):
            self.join_str = ","
        if self.operator == "#":
            self.start = "#"
        if self.operator == "?":
            self.start = "?"
            self.join_str = "&"

        if self.operator in ("+", "#"):
            self.safe = URIVariable.reserved

    def _query_expansion(
        self,
        name: str,
        value: VariableValue,
        explode: bool,
        prefix: t.Optional[int],
    ) -> t.Optional[str]:
        """Expansion method for the '?' and '&' operators."""
        if value is None:
            return None

        tuples, items = is_list_of_tuples(value)

        safe = self.safe
        if list_test(value) and not tuples:
            if not value:
                return None
            value = t.cast(t.Sequence[ScalarVariableValue], value)
            if explode:
                return self.join_str.join(
                    f"{name}={quote(v, safe)}" for v in value
                )
            else:
                value = ",".join(quote(v, safe) for v in value)
                return f"{name}={value}"

        if dict_test(value) or tuples:
            if not value:
                return None
            value = t.cast(t.Mapping[str, ScalarVariableValue], value)
            items = items or sorted(value.items())
            if explode:
                return self.join_str.join(
                    f"{quote(k, safe)}={quote(v, safe)}" for k, v in items
                )
            else:
                value = ",".join(
                    f"{quote(k, safe)},{quote(v, safe)}" for k, v in items
                )
                return f"{name}={value}"

        if value:
            value = t.cast(t.Text, value)
            value = value[:prefix] if prefix else value
            return f"{name}={quote(value, safe)}"
        return name + "="

    def _label_path_expansion(
        self,
        name: str,
        value: VariableValue,
        explode: bool,
        prefix: t.Optional[int],
    ) -> t.Optional[str]:
        """Label and path expansion method.

        Expands for operators: '/', '.'

        """
        join_str = self.join_str
        safe = self.safe

        if value is None or (
            not isinstance(value, (str, int, float, complex))
            and len(value) == 0
        ):
            return None

        tuples, items = is_list_of_tuples(value)

        if list_test(value) and not tuples:
            if not explode:
                join_str = ","

            value = t.cast(t.Sequence[ScalarVariableValue], value)
            fragments = [quote(v, safe) for v in value if v is not None]
            return join_str.join(fragments) if fragments else None

        if dict_test(value) or tuples:
            value = t.cast(t.Mapping[str, ScalarVariableValue], value)
            items = items or sorted(value.items())
            format_str = "%s=%s"
            if not explode:
                format_str = "%s,%s"
                join_str = ","

            expanded = join_str.join(
                format_str % (quote(k, safe), quote(v, safe))
                for k, v in items
                if v is not None
            )
            return expanded if expanded else None

        value = t.cast(t.Text, value)
        value = value[:prefix] if prefix else value
        return quote(value, safe)

    def _semi_path_expansion(
        self,
        name: str,
        value: VariableValue,
        explode: bool,
        prefix: t.Optional[int],
    ) -> t.Optional[str]:
        """Expansion method for ';' operator."""
        join_str = self.join_str
        safe = self.safe

        if value is None:
            return None

        if self.operator == "?":
            join_str = "&"

        tuples, items = is_list_of_tuples(value)

        if list_test(value) and not tuples:
            value = t.cast(t.Sequence[ScalarVariableValue], value)
            if explode:
                expanded = join_str.join(
                    f"{name}={quote(v, safe)}" for v in value if v is not None
                )
                return expanded if expanded else None
            else:
                value = ",".join(quote(v, safe) for v in value)
                return f"{name}={value}"

        if dict_test(value) or tuples:
            value = t.cast(t.Mapping[str, ScalarVariableValue], value)
            items = items or sorted(value.items())

            if explode:
                return join_str.join(
                    f"{quote(k, safe)}={quote(v, safe)}"
                    for k, v in items
                    if v is not None
                )
            else:
                expanded = ",".join(
                    f"{quote(k, safe)},{quote(v, safe)}"
                    for k, v in items
                    if v is not None
                )
                return f"{name}={expanded}"

        value = t.cast(t.Text, value)
        value = value[:prefix] if prefix else value
        if value:
            return f"{name}={quote(value, safe)}"

        return name

    def _string_expansion(
        self,
        name: str,
        value: VariableValue,
        explode: bool,
        prefix: t.Optional[int],
    ) -> t.Optional[str]:
        if value is None:
            return None

        tuples, items = is_list_of_tuples(value)

        if list_test(value) and not tuples:
            value = t.cast(t.Sequence[ScalarVariableValue], value)
            return ",".join(quote(v, self.safe) for v in value)

        if dict_test(value) or tuples:
            value = t.cast(t.Mapping[str, ScalarVariableValue], value)
            items = items or sorted(value.items())
            format_str = "%s=%s" if explode else "%s,%s"

            return ",".join(
                format_str % (quote(k, self.safe), quote(v, self.safe))
                for k, v in items
            )

        value = t.cast(t.Text, value)
        value = value[:prefix] if prefix else value
        return quote(value, self.safe)

    def expand(
        self, var_dict: t.Optional[VariableValueDict] = None
    ) -> t.Mapping[str, str]:
        """Expand the variable in question.

        Using ``var_dict`` and the previously parsed defaults, expand this
        variable and subvariables.

        :param dict var_dict: dictionary of key-value pairs to be used during
            expansion
        :returns: dict(variable=value)

        Examples::

            # (1)
            v = URIVariable('/var')
            expansion = v.expand({'var': 'value'})
            print(expansion)
            # => {'/var': '/value'}

            # (2)
            v = URIVariable('?var,hello,x,y')
            expansion = v.expand({'var': 'value', 'hello': 'Hello World!',
                                  'x': '1024', 'y': '768'})
            print(expansion)
            # => {'?var,hello,x,y':
            #     '?var=value&hello=Hello%20World%21&x=1024&y=768'}

        """
        return_values = []
        if var_dict is None:
            return {self.original: self.original}

        for name, opts in self.variables:
            value = var_dict.get(name, None)
            if not value and value != "" and name in self.defaults:
                value = self.defaults[name]

            if value is None:
                continue

            expanded = None
            if self.operator in ("/", "."):
                expansion = self._label_path_expansion
            elif self.operator in ("?", "&"):
                expansion = self._query_expansion
            elif self.operator == ";":
                expansion = self._semi_path_expansion
            else:
                expansion = self._string_expansion

            expanded = expansion(name, value, opts["explode"], opts["prefix"])

            if expanded is not None:
                return_values.append(expanded)

        value = ""
        if return_values:
            value = self.start + self.join_str.join(return_values)
        return {self.original: value}


def is_list_of_tuples(
    value: t.Any,
) -> t.Tuple[bool, t.Optional[t.Sequence[t.Tuple[str, ScalarVariableValue]]]]:
    if (
        not value
        or not isinstance(value, (list, tuple))
        or not all(isinstance(t, tuple) and len(t) == 2 for t in value)
    ):
        return False, None

    return True, value


def list_test(value: t.Any) -> bool:
    return isinstance(value, (list, tuple))


def dict_test(value: t.Any) -> bool:
    return isinstance(value, (dict, collections.abc.MutableMapping))


def _encode(value: t.AnyStr, encoding: str = "utf-8") -> bytes:
    if isinstance(value, str):
        return value.encode(encoding)
    return value


def quote(value: t.Any, safe: str) -> str:
    if not isinstance(value, (str, bytes)):
        value = str(value)
    return urllib.parse.quote(_encode(value), safe)
