# -*- coding: utf-8 -*-
"""
    cssselect.parser
    ================

    Tokenizer, parser and parsed objects for CSS selectors.


    :copyright: (c) 2007-2012 Ian Bicking and contributors.
                See AUTHORS for more details.
    :license: BSD, see LICENSE for more details.

"""

import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union


def ascii_lower(string: str) -> str:
    """Lower-case, but only in the ASCII range."""
    return string.encode("utf8").lower().decode("utf8")


class SelectorError(Exception):
    """Common parent for :class:`SelectorSyntaxError` and
    :class:`ExpressionError`.

    You can just use ``except SelectorError:`` when calling
    :meth:`~GenericTranslator.css_to_xpath` and handle both exceptions types.

    """


class SelectorSyntaxError(SelectorError, SyntaxError):
    """Parsing a selector that does not match the grammar."""


#### Parsed objects

Tree = Union[
    "Element",
    "Hash",
    "Class",
    "Function",
    "Pseudo",
    "Attrib",
    "Negation",
    "Relation",
    "Matching",
    "SpecificityAdjustment",
    "CombinedSelector",
]
PseudoElement = Union["FunctionalPseudoElement", str]


class Selector:
    """
    Represents a parsed selector.

    :meth:`~GenericTranslator.selector_to_xpath` accepts this object,
    but ignores :attr:`pseudo_element`. It is the user’s responsibility
    to account for pseudo-elements and reject selectors with unknown
    or unsupported pseudo-elements.

    """

    def __init__(self, tree: Tree, pseudo_element: Optional[PseudoElement] = None) -> None:
        self.parsed_tree = tree
        if pseudo_element is not None and not isinstance(pseudo_element, FunctionalPseudoElement):
            pseudo_element = ascii_lower(pseudo_element)
        #: A :class:`FunctionalPseudoElement`,
        #: or the identifier for the pseudo-element as a string,
        #  or ``None``.
        #:
        #: +-------------------------+----------------+--------------------------------+
        #: |                         | Selector       | Pseudo-element                 |
        #: +=========================+================+================================+
        #: | CSS3 syntax             | ``a::before``  | ``'before'``                   |
        #: +-------------------------+----------------+--------------------------------+
        #: | Older syntax            | ``a:before``   | ``'before'``                   |
        #: +-------------------------+----------------+--------------------------------+
        #: | From the Lists3_ draft, | ``li::marker`` | ``'marker'``                   |
        #: | not in Selectors3       |                |                                |
        #: +-------------------------+----------------+--------------------------------+
        #: | Invalid pseudo-class    | ``li:marker``  | ``None``                       |
        #: +-------------------------+----------------+--------------------------------+
        #: | Functional              | ``a::foo(2)``  | ``FunctionalPseudoElement(…)`` |
        #: +-------------------------+----------------+--------------------------------+
        #:
        #: .. _Lists3: http://www.w3.org/TR/2011/WD-css3-lists-20110524/#marker-pseudoelement
        self.pseudo_element = pseudo_element

    def __repr__(self) -> str:
        if isinstance(self.pseudo_element, FunctionalPseudoElement):
            pseudo_element = repr(self.pseudo_element)
        elif self.pseudo_element:
            pseudo_element = "::%s" % self.pseudo_element
        else:
            pseudo_element = ""
        return "%s[%r%s]" % (self.__class__.__name__, self.parsed_tree, pseudo_element)

    def canonical(self) -> str:
        """Return a CSS representation for this selector (a string)"""
        if isinstance(self.pseudo_element, FunctionalPseudoElement):
            pseudo_element = "::%s" % self.pseudo_element.canonical()
        elif self.pseudo_element:
            pseudo_element = "::%s" % self.pseudo_element
        else:
            pseudo_element = ""
        res = "%s%s" % (self.parsed_tree.canonical(), pseudo_element)
        if len(res) > 1:
            res = res.lstrip("*")
        return res

    def specificity(self) -> Tuple[int, int, int]:
        """Return the specificity_ of this selector as a tuple of 3 integers.

        .. _specificity: http://www.w3.org/TR/selectors/#specificity

        """
        a, b, c = self.parsed_tree.specificity()
        if self.pseudo_element:
            c += 1
        return a, b, c


class Class:
    """
    Represents selector.class_name
    """

    def __init__(self, selector: Tree, class_name: str) -> None:
        self.selector = selector
        self.class_name = class_name

    def __repr__(self) -> str:
        return "%s[%r.%s]" % (self.__class__.__name__, self.selector, self.class_name)

    def canonical(self) -> str:
        return "%s.%s" % (self.selector.canonical(), self.class_name)

    def specificity(self) -> Tuple[int, int, int]:
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class FunctionalPseudoElement:
    """
    Represents selector::name(arguments)

    .. attribute:: name

        The name (identifier) of the pseudo-element, as a string.

    .. attribute:: arguments

        The arguments of the pseudo-element, as a list of tokens.

        **Note:** tokens are not part of the public API,
        and may change between cssselect versions.
        Use at your own risks.

    """

    def __init__(self, name: str, arguments: Sequence["Token"]):
        self.name = ascii_lower(name)
        self.arguments = arguments

    def __repr__(self) -> str:
        return "%s[::%s(%r)]" % (
            self.__class__.__name__,
            self.name,
            [token.value for token in self.arguments],
        )

    def argument_types(self) -> List[str]:
        return [token.type for token in self.arguments]

    def canonical(self) -> str:
        args = "".join(token.css() for token in self.arguments)
        return "%s(%s)" % (self.name, args)


class Function:
    """
    Represents selector:name(expr)
    """

    def __init__(self, selector: Tree, name: str, arguments: Sequence["Token"]) -> None:
        self.selector = selector
        self.name = ascii_lower(name)
        self.arguments = arguments

    def __repr__(self) -> str:
        return "%s[%r:%s(%r)]" % (
            self.__class__.__name__,
            self.selector,
            self.name,
            [token.value for token in self.arguments],
        )

    def argument_types(self) -> List[str]:
        return [token.type for token in self.arguments]

    def canonical(self) -> str:
        args = "".join(token.css() for token in self.arguments)
        return "%s:%s(%s)" % (self.selector.canonical(), self.name, args)

    def specificity(self) -> Tuple[int, int, int]:
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class Pseudo:
    """
    Represents selector:ident
    """

    def __init__(self, selector: Tree, ident: str) -> None:
        self.selector = selector
        self.ident = ascii_lower(ident)

    def __repr__(self) -> str:
        return "%s[%r:%s]" % (self.__class__.__name__, self.selector, self.ident)

    def canonical(self) -> str:
        return "%s:%s" % (self.selector.canonical(), self.ident)

    def specificity(self) -> Tuple[int, int, int]:
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class Negation:
    """
    Represents selector:not(subselector)
    """

    def __init__(self, selector: Tree, subselector: Tree) -> None:
        self.selector = selector
        self.subselector = subselector

    def __repr__(self) -> str:
        return "%s[%r:not(%r)]" % (self.__class__.__name__, self.selector, self.subselector)

    def canonical(self) -> str:
        subsel = self.subselector.canonical()
        if len(subsel) > 1:
            subsel = subsel.lstrip("*")
        return "%s:not(%s)" % (self.selector.canonical(), subsel)

    def specificity(self) -> Tuple[int, int, int]:
        a1, b1, c1 = self.selector.specificity()
        a2, b2, c2 = self.subselector.specificity()
        return a1 + a2, b1 + b2, c1 + c2


class Relation:
    """
    Represents selector:has(subselector)
    """

    def __init__(self, selector: Tree, combinator: "Token", subselector: Selector):
        self.selector = selector
        self.combinator = combinator
        self.subselector = subselector

    def __repr__(self) -> str:
        return "%s[%r:has(%r)]" % (
            self.__class__.__name__,
            self.selector,
            self.subselector,
        )

    def canonical(self) -> str:
        try:
            subsel = self.subselector[0].canonical()  # type: ignore
        except TypeError:
            subsel = self.subselector.canonical()
        if len(subsel) > 1:
            subsel = subsel.lstrip("*")
        return "%s:has(%s)" % (self.selector.canonical(), subsel)

    def specificity(self) -> Tuple[int, int, int]:
        a1, b1, c1 = self.selector.specificity()
        try:
            a2, b2, c2 = self.subselector[-1].specificity()  # type: ignore
        except TypeError:
            a2, b2, c2 = self.subselector.specificity()
        return a1 + a2, b1 + b2, c1 + c2


class Matching:
    """
    Represents selector:is(selector_list)
    """

    def __init__(self, selector: Tree, selector_list: Iterable[Tree]):
        self.selector = selector
        self.selector_list = selector_list

    def __repr__(self) -> str:
        return "%s[%r:is(%s)]" % (
            self.__class__.__name__,
            self.selector,
            ", ".join(map(repr, self.selector_list)),
        )

    def canonical(self) -> str:
        selector_arguments = []
        for s in self.selector_list:
            selarg = s.canonical()
            selector_arguments.append(selarg.lstrip("*"))
        return "%s:is(%s)" % (self.selector.canonical(), ", ".join(map(str, selector_arguments)))

    def specificity(self) -> Tuple[int, int, int]:
        return max(x.specificity() for x in self.selector_list)


class SpecificityAdjustment:
    """
    Represents selector:where(selector_list)
    Same as selector:is(selector_list), but its specificity is always 0
    """

    def __init__(self, selector: Tree, selector_list: List[Tree]):
        self.selector = selector
        self.selector_list = selector_list

    def __repr__(self) -> str:
        return "%s[%r:where(%s)]" % (
            self.__class__.__name__,
            self.selector,
            ", ".join(map(repr, self.selector_list)),
        )

    def canonical(self) -> str:
        selector_arguments = []
        for s in self.selector_list:
            selarg = s.canonical()
            selector_arguments.append(selarg.lstrip("*"))
        return "%s:where(%s)" % (
            self.selector.canonical(),
            ", ".join(map(str, selector_arguments)),
        )

    def specificity(self) -> Tuple[int, int, int]:
        return 0, 0, 0


class Attrib:
    """
    Represents selector[namespace|attrib operator value]
    """

    @typing.overload
    def __init__(
        self,
        selector: Tree,
        namespace: Optional[str],
        attrib: str,
        operator: 'typing.Literal["exists"]',
        value: None,
    ) -> None:
        ...

    @typing.overload
    def __init__(
        self, selector: Tree, namespace: Optional[str], attrib: str, operator: str, value: "Token"
    ) -> None:
        ...

    def __init__(
        self,
        selector: Tree,
        namespace: Optional[str],
        attrib: str,
        operator: str,
        value: Optional["Token"],
    ) -> None:
        self.selector = selector
        self.namespace = namespace
        self.attrib = attrib
        self.operator = operator
        self.value = value

    def __repr__(self) -> str:
        if self.namespace:
            attrib = "%s|%s" % (self.namespace, self.attrib)
        else:
            attrib = self.attrib
        if self.operator == "exists":
            return "%s[%r[%s]]" % (self.__class__.__name__, self.selector, attrib)
        else:
            return "%s[%r[%s %s %r]]" % (
                self.__class__.__name__,
                self.selector,
                attrib,
                self.operator,
                typing.cast("Token", self.value).value,
            )

    def canonical(self) -> str:
        if self.namespace:
            attrib = "%s|%s" % (self.namespace, self.attrib)
        else:
            attrib = self.attrib

        if self.operator == "exists":
            op = attrib
        else:
            op = "%s%s%s" % (attrib, self.operator, typing.cast("Token", self.value).css())

        return "%s[%s]" % (self.selector.canonical(), op)

    def specificity(self) -> Tuple[int, int, int]:
        a, b, c = self.selector.specificity()
        b += 1
        return a, b, c


class Element:
    """
    Represents namespace|element

    `None` is for the universal selector '*'

    """

    def __init__(self, namespace: Optional[str] = None, element: Optional[str] = None) -> None:
        self.namespace = namespace
        self.element = element

    def __repr__(self) -> str:
        return "%s[%s]" % (self.__class__.__name__, self.canonical())

    def canonical(self) -> str:
        element = self.element or "*"
        if self.namespace:
            element = "%s|%s" % (self.namespace, element)
        return element

    def specificity(self) -> Tuple[int, int, int]:
        if self.element:
            return 0, 0, 1
        else:
            return 0, 0, 0


class Hash:
    """
    Represents selector#id
    """

    def __init__(self, selector: Tree, id: str) -> None:
        self.selector = selector
        self.id = id

    def __repr__(self) -> str:
        return "%s[%r#%s]" % (self.__class__.__name__, self.selector, self.id)

    def canonical(self) -> str:
        return "%s#%s" % (self.selector.canonical(), self.id)

    def specificity(self) -> Tuple[int, int, int]:
        a, b, c = self.selector.specificity()
        a += 1
        return a, b, c


class CombinedSelector:
    def __init__(self, selector: Tree, combinator: str, subselector: Tree) -> None:
        assert selector is not None
        self.selector = selector
        self.combinator = combinator
        self.subselector = subselector

    def __repr__(self) -> str:
        if self.combinator == " ":
            comb = "<followed>"
        else:
            comb = self.combinator
        return "%s[%r %s %r]" % (self.__class__.__name__, self.selector, comb, self.subselector)

    def canonical(self) -> str:
        subsel = self.subselector.canonical()
        if len(subsel) > 1:
            subsel = subsel.lstrip("*")
        return "%s %s %s" % (self.selector.canonical(), self.combinator, subsel)

    def specificity(self) -> Tuple[int, int, int]:
        a1, b1, c1 = self.selector.specificity()
        a2, b2, c2 = self.subselector.specificity()
        return a1 + a2, b1 + b2, c1 + c2


#### Parser

# foo
_el_re = re.compile(r"^[ \t\r\n\f]*([a-zA-Z]+)[ \t\r\n\f]*$")

# foo#bar or #bar
_id_re = re.compile(r"^[ \t\r\n\f]*([a-zA-Z]*)#([a-zA-Z0-9_-]+)[ \t\r\n\f]*$")

# foo.bar or .bar
_class_re = re.compile(r"^[ \t\r\n\f]*([a-zA-Z]*)\.([a-zA-Z][a-zA-Z0-9_-]*)[ \t\r\n\f]*$")


def parse(css: str) -> List[Selector]:
    """Parse a CSS *group of selectors*.

    If you don't care about pseudo-elements or selector specificity,
    you can skip this and use :meth:`~GenericTranslator.css_to_xpath`.

    :param css:
        A *group of selectors* as a string.
    :raises:
        :class:`SelectorSyntaxError` on invalid selectors.
    :returns:
        A list of parsed :class:`Selector` objects, one for each
        selector in the comma-separated group.

    """
    # Fast path for simple cases
    match = _el_re.match(css)
    if match:
        return [Selector(Element(element=match.group(1)))]
    match = _id_re.match(css)
    if match is not None:
        return [Selector(Hash(Element(element=match.group(1) or None), match.group(2)))]
    match = _class_re.match(css)
    if match is not None:
        return [Selector(Class(Element(element=match.group(1) or None), match.group(2)))]

    stream = TokenStream(tokenize(css))
    stream.source = css
    return list(parse_selector_group(stream))


#    except SelectorSyntaxError:
#        e = sys.exc_info()[1]
#        message = "%s at %s -> %r" % (
#            e, stream.used, stream.peek())
#        e.msg = message
#        e.args = tuple([message])
#        raise


def parse_selector_group(stream: "TokenStream") -> Iterator[Selector]:
    stream.skip_whitespace()
    while 1:
        yield Selector(*parse_selector(stream))
        if stream.peek() == ("DELIM", ","):
            stream.next()
            stream.skip_whitespace()
        else:
            break


def parse_selector(stream: "TokenStream") -> Tuple[Tree, Optional[PseudoElement]]:
    result, pseudo_element = parse_simple_selector(stream)
    while 1:
        stream.skip_whitespace()
        peek = stream.peek()
        if peek in (("EOF", None), ("DELIM", ",")):
            break
        if pseudo_element:
            raise SelectorSyntaxError(
                "Got pseudo-element ::%s not at the end of a selector" % pseudo_element
            )
        if peek.is_delim("+", ">", "~"):
            # A combinator
            combinator = typing.cast(str, stream.next().value)
            stream.skip_whitespace()
        else:
            # By exclusion, the last parse_simple_selector() ended
            # at peek == ' '
            combinator = " "
        next_selector, pseudo_element = parse_simple_selector(stream)
        result = CombinedSelector(result, combinator, next_selector)
    return result, pseudo_element


def parse_simple_selector(
    stream: "TokenStream", inside_negation: bool = False
) -> Tuple[Tree, Optional[PseudoElement]]:
    stream.skip_whitespace()
    selector_start = len(stream.used)
    peek = stream.peek()
    if peek.type == "IDENT" or peek == ("DELIM", "*"):
        if peek.type == "IDENT":
            namespace = stream.next().value
        else:
            stream.next()
            namespace = None
        if stream.peek() == ("DELIM", "|"):
            stream.next()
            element = stream.next_ident_or_star()
        else:
            element = namespace
            namespace = None
    else:
        element = namespace = None
    result: Tree = Element(namespace, element)
    pseudo_element: Optional[PseudoElement] = None
    while 1:
        peek = stream.peek()
        if (
            peek.type in ("S", "EOF")
            or peek.is_delim(",", "+", ">", "~")
            or (inside_negation and peek == ("DELIM", ")"))
        ):
            break
        if pseudo_element:
            raise SelectorSyntaxError(
                "Got pseudo-element ::%s not at the end of a selector" % pseudo_element
            )
        if peek.type == "HASH":
            result = Hash(result, typing.cast(str, stream.next().value))
        elif peek == ("DELIM", "."):
            stream.next()
            result = Class(result, stream.next_ident())
        elif peek == ("DELIM", "|"):
            stream.next()
            result = Element(None, stream.next_ident())
        elif peek == ("DELIM", "["):
            stream.next()
            result = parse_attrib(result, stream)
        elif peek == ("DELIM", ":"):
            stream.next()
            if stream.peek() == ("DELIM", ":"):
                stream.next()
                pseudo_element = stream.next_ident()
                if stream.peek() == ("DELIM", "("):
                    stream.next()
                    pseudo_element = FunctionalPseudoElement(
                        pseudo_element, parse_arguments(stream)
                    )
                continue
            ident = stream.next_ident()
            if ident.lower() in ("first-line", "first-letter", "before", "after"):
                # Special case: CSS 2.1 pseudo-elements can have a single ':'
                # Any new pseudo-element must have two.
                pseudo_element = str(ident)
                continue
            if stream.peek() != ("DELIM", "("):
                result = Pseudo(result, ident)
                if repr(result) == "Pseudo[Element[*]:scope]":
                    if not (
                        len(stream.used) == 2
                        or (len(stream.used) == 3 and stream.used[0].type == "S")
                        or (len(stream.used) >= 3 and stream.used[-3].is_delim(","))
                        or (
                            len(stream.used) >= 4
                            and stream.used[-3].type == "S"
                            and stream.used[-4].is_delim(",")
                        )
                    ):
                        raise SelectorSyntaxError(
                            'Got immediate child pseudo-element ":scope" '
                            "not at the start of a selector"
                        )
                continue
            stream.next()
            stream.skip_whitespace()
            if ident.lower() == "not":
                if inside_negation:
                    raise SelectorSyntaxError("Got nested :not()")
                argument, argument_pseudo_element = parse_simple_selector(
                    stream, inside_negation=True
                )
                next = stream.next()
                if argument_pseudo_element:
                    raise SelectorSyntaxError(
                        "Got pseudo-element ::%s inside :not() at %s"
                        % (argument_pseudo_element, next.pos)
                    )
                if next != ("DELIM", ")"):
                    raise SelectorSyntaxError("Expected ')', got %s" % (next,))
                result = Negation(result, argument)
            elif ident.lower() == "has":
                combinator, arguments = parse_relative_selector(stream)
                result = Relation(result, combinator, arguments)

            elif ident.lower() in ("matches", "is"):
                selectors = parse_simple_selector_arguments(stream)
                result = Matching(result, selectors)
            elif ident.lower() == "where":
                selectors = parse_simple_selector_arguments(stream)
                result = SpecificityAdjustment(result, selectors)
            else:
                result = Function(result, ident, parse_arguments(stream))
        else:
            raise SelectorSyntaxError("Expected selector, got %s" % (peek,))
    if len(stream.used) == selector_start:
        raise SelectorSyntaxError("Expected selector, got %s" % (stream.peek(),))
    return result, pseudo_element


def parse_arguments(stream: "TokenStream") -> List["Token"]:
    arguments: List["Token"] = []
    while 1:
        stream.skip_whitespace()
        next = stream.next()
        if next.type in ("IDENT", "STRING", "NUMBER") or next in [("DELIM", "+"), ("DELIM", "-")]:
            arguments.append(next)
        elif next == ("DELIM", ")"):
            return arguments
        else:
            raise SelectorSyntaxError("Expected an argument, got %s" % (next,))


def parse_relative_selector(stream: "TokenStream") -> Tuple["Token", Selector]:
    stream.skip_whitespace()
    subselector = ""
    next = stream.next()

    if next in [("DELIM", "+"), ("DELIM", "-"), ("DELIM", ">"), ("DELIM", "~")]:
        combinator = next
        stream.skip_whitespace()
        next = stream.next()
    else:
        combinator = Token("DELIM", " ", pos=0)

    while 1:
        if next.type in ("IDENT", "STRING", "NUMBER") or next in [("DELIM", "."), ("DELIM", "*")]:
            subselector += typing.cast(str, next.value)
        elif next == ("DELIM", ")"):
            result = parse(subselector)
            return combinator, result[0]
        else:
            raise SelectorSyntaxError("Expected an argument, got %s" % (next,))
        next = stream.next()


def parse_simple_selector_arguments(stream: "TokenStream") -> List[Tree]:
    arguments = []
    while 1:
        result, pseudo_element = parse_simple_selector(stream, True)
        if pseudo_element:
            raise SelectorSyntaxError(
                "Got pseudo-element ::%s inside function" % (pseudo_element,)
            )
        stream.skip_whitespace()
        next = stream.next()
        if next in (("EOF", None), ("DELIM", ",")):
            stream.next()
            stream.skip_whitespace()
            arguments.append(result)
        elif next == ("DELIM", ")"):
            arguments.append(result)
            break
        else:
            raise SelectorSyntaxError("Expected an argument, got %s" % (next,))
    return arguments


def parse_attrib(selector: Tree, stream: "TokenStream") -> Attrib:
    stream.skip_whitespace()
    attrib = stream.next_ident_or_star()
    if attrib is None and stream.peek() != ("DELIM", "|"):
        raise SelectorSyntaxError("Expected '|', got %s" % (stream.peek(),))
    namespace: Optional[str]
    op: Optional[str]
    if stream.peek() == ("DELIM", "|"):
        stream.next()
        if stream.peek() == ("DELIM", "="):
            namespace = None
            stream.next()
            op = "|="
        else:
            namespace = attrib
            attrib = stream.next_ident()
            op = None
    else:
        namespace = op = None
    if op is None:
        stream.skip_whitespace()
        next = stream.next()
        if next == ("DELIM", "]"):
            return Attrib(selector, namespace, typing.cast(str, attrib), "exists", None)
        elif next == ("DELIM", "="):
            op = "="
        elif next.is_delim("^", "$", "*", "~", "|", "!") and (stream.peek() == ("DELIM", "=")):
            op = typing.cast(str, next.value) + "="
            stream.next()
        else:
            raise SelectorSyntaxError("Operator expected, got %s" % (next,))
    stream.skip_whitespace()
    value = stream.next()
    if value.type not in ("IDENT", "STRING"):
        raise SelectorSyntaxError("Expected string or ident, got %s" % (value,))
    stream.skip_whitespace()
    next = stream.next()
    if next != ("DELIM", "]"):
        raise SelectorSyntaxError("Expected ']', got %s" % (next,))
    return Attrib(selector, namespace, typing.cast(str, attrib), op, value)


def parse_series(tokens: Iterable["Token"]) -> Tuple[int, int]:
    """
    Parses the arguments for :nth-child() and friends.

    :raises: A list of tokens
    :returns: :``(a, b)``

    """
    for token in tokens:
        if token.type == "STRING":
            raise ValueError("String tokens not allowed in series.")
    s = "".join(typing.cast(str, token.value) for token in tokens).strip()
    if s == "odd":
        return 2, 1
    elif s == "even":
        return 2, 0
    elif s == "n":
        return 1, 0
    if "n" not in s:
        # Just b
        return 0, int(s)
    a, b = s.split("n", 1)
    a_as_int: int
    if not a:
        a_as_int = 1
    elif a == "-" or a == "+":
        a_as_int = int(a + "1")
    else:
        a_as_int = int(a)
    b_as_int: int
    if not b:
        b_as_int = 0
    else:
        b_as_int = int(b)
    return a_as_int, b_as_int


#### Token objects


class Token(Tuple[str, Optional[str]]):
    @typing.overload
    def __new__(
        cls,
        type_: 'typing.Literal["IDENT", "HASH", "STRING", "S", "DELIM", "NUMBER"]',
        value: str,
        pos: int,
    ) -> "Token":
        ...

    @typing.overload
    def __new__(cls, type_: 'typing.Literal["EOF"]', value: None, pos: int) -> "Token":
        ...

    def __new__(cls, type_: str, value: Optional[str], pos: int) -> "Token":
        obj = tuple.__new__(cls, (type_, value))
        obj.pos = pos
        return obj

    def __repr__(self) -> str:
        return "<%s '%s' at %i>" % (self.type, self.value, self.pos)

    def is_delim(self, *values: str) -> bool:
        return self.type == "DELIM" and self.value in values

    pos: int

    @property
    def type(self) -> str:
        return self[0]

    @property
    def value(self) -> Optional[str]:
        return self[1]

    def css(self) -> str:
        if self.type == "STRING":
            return repr(self.value)
        else:
            return typing.cast(str, self.value)


class EOFToken(Token):
    def __new__(cls, pos: int) -> "EOFToken":
        return typing.cast("EOFToken", Token.__new__(cls, "EOF", None, pos))

    def __repr__(self) -> str:
        return "<%s at %i>" % (self.type, self.pos)


#### Tokenizer


class TokenMacros:
    unicode_escape = r"\\([0-9a-f]{1,6})(?:\r\n|[ \n\r\t\f])?"
    escape = unicode_escape + r"|\\[^\n\r\f0-9a-f]"
    string_escape = r"\\(?:\n|\r\n|\r|\f)|" + escape
    nonascii = r"[^\0-\177]"
    nmchar = "[_a-z0-9-]|%s|%s" % (escape, nonascii)
    nmstart = "[_a-z]|%s|%s" % (escape, nonascii)


if typing.TYPE_CHECKING:

    class MatchFunc(typing.Protocol):
        def __call__(
            self, string: str, pos: int = ..., endpos: int = ...
        ) -> Optional["re.Match[str]"]:
            ...


def _compile(pattern: str) -> "MatchFunc":
    return re.compile(pattern % vars(TokenMacros), re.IGNORECASE).match


_match_whitespace = _compile(r"[ \t\r\n\f]+")
_match_number = _compile(r"[+-]?(?:[0-9]*\.[0-9]+|[0-9]+)")
_match_hash = _compile("#(?:%(nmchar)s)+")
_match_ident = _compile("-?(?:%(nmstart)s)(?:%(nmchar)s)*")
_match_string_by_quote = {
    "'": _compile(r"([^\n\r\f\\']|%(string_escape)s)*"),
    '"': _compile(r'([^\n\r\f\\"]|%(string_escape)s)*'),
}

_sub_simple_escape = re.compile(r"\\(.)").sub
_sub_unicode_escape = re.compile(TokenMacros.unicode_escape, re.I).sub
_sub_newline_escape = re.compile(r"\\(?:\n|\r\n|\r|\f)").sub

# Same as r'\1', but faster on CPython
_replace_simple = operator.methodcaller("group", 1)


def _replace_unicode(match: "re.Match[str]") -> str:
    codepoint = int(match.group(1), 16)
    if codepoint > sys.maxunicode:
        codepoint = 0xFFFD
    return chr(codepoint)


def unescape_ident(value: str) -> str:
    value = _sub_unicode_escape(_replace_unicode, value)
    value = _sub_simple_escape(_replace_simple, value)
    return value


def tokenize(s: str) -> Iterator[Token]:
    pos = 0
    len_s = len(s)
    while pos < len_s:
        match = _match_whitespace(s, pos=pos)
        if match:
            yield Token("S", " ", pos)
            pos = match.end()
            continue

        match = _match_ident(s, pos=pos)
        if match:
            value = _sub_simple_escape(
                _replace_simple, _sub_unicode_escape(_replace_unicode, match.group())
            )
            yield Token("IDENT", value, pos)
            pos = match.end()
            continue

        match = _match_hash(s, pos=pos)
        if match:
            value = _sub_simple_escape(
                _replace_simple, _sub_unicode_escape(_replace_unicode, match.group()[1:])
            )
            yield Token("HASH", value, pos)
            pos = match.end()
            continue

        quote = s[pos]
        if quote in _match_string_by_quote:
            match = _match_string_by_quote[quote](s, pos=pos + 1)
            assert match, "Should have found at least an empty match"
            end_pos = match.end()
            if end_pos == len_s:
                raise SelectorSyntaxError("Unclosed string at %s" % pos)
            if s[end_pos] != quote:
                raise SelectorSyntaxError("Invalid string at %s" % pos)
            value = _sub_simple_escape(
                _replace_simple,
                _sub_unicode_escape(_replace_unicode, _sub_newline_escape("", match.group())),
            )
            yield Token("STRING", value, pos)
            pos = end_pos + 1
            continue

        match = _match_number(s, pos=pos)
        if match:
            value = match.group()
            yield Token("NUMBER", value, pos)
            pos = match.end()
            continue

        pos2 = pos + 2
        if s[pos:pos2] == "/*":
            pos = s.find("*/", pos2)
            if pos == -1:
                pos = len_s
            else:
                pos += 2
            continue

        yield Token("DELIM", s[pos], pos)
        pos += 1

    assert pos == len_s
    yield EOFToken(pos)


class TokenStream:
    def __init__(self, tokens: Iterable[Token], source: Optional[str] = None) -> None:
        self.used: List[Token] = []
        self.tokens = iter(tokens)
        self.source = source
        self.peeked: Optional[Token] = None
        self._peeking = False
        self.next_token = self.tokens.__next__

    def next(self) -> Token:
        if self._peeking:
            self._peeking = False
            self.used.append(typing.cast(Token, self.peeked))
            return typing.cast(Token, self.peeked)
        else:
            next = self.next_token()
            self.used.append(next)
            return next

    def peek(self) -> Token:
        if not self._peeking:
            self.peeked = self.next_token()
            self._peeking = True
        return typing.cast(Token, self.peeked)

    def next_ident(self) -> str:
        next = self.next()
        if next.type != "IDENT":
            raise SelectorSyntaxError("Expected ident, got %s" % (next,))
        return typing.cast(str, next.value)

    def next_ident_or_star(self) -> Optional[str]:
        next = self.next()
        if next.type == "IDENT":
            return next.value
        elif next == ("DELIM", "*"):
            return None
        else:
            raise SelectorSyntaxError("Expected ident or '*', got %s" % (next,))

    def skip_whitespace(self) -> None:
        peek = self.peek()
        if peek.type == "S":
            self.next()
