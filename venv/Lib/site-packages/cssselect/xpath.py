# -*- coding: utf-8 -*-
"""
    cssselect.xpath
    ===============

    Translation of parsed CSS selectors to XPath expressions.


    :copyright: (c) 2007-2012 Ian Bicking and contributors.
                See AUTHORS for more details.
    :license: BSD, see LICENSE for more details.

"""

import re
import typing
import warnings
from typing import Optional

from cssselect.parser import (
    parse,
    parse_series,
    PseudoElement,
    Selector,
    SelectorError,
    Tree,
    Element,
    Hash,
    Class,
    Function,
    Pseudo,
    Attrib,
    Negation,
    Relation,
    Matching,
    SpecificityAdjustment,
    CombinedSelector,
)


@typing.no_type_check
def _unicode_safe_getattr(obj, name, default=None):
    warnings.warn(
        "_unicode_safe_getattr is deprecated and will be removed in the"
        " next release, use getattr() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(obj, name, default)


class ExpressionError(SelectorError, RuntimeError):
    """Unknown or unsupported selector (eg. pseudo-class)."""


#### XPath Helpers


class XPathExpr:
    def __init__(
        self, path: str = "", element: str = "*", condition: str = "", star_prefix: bool = False
    ) -> None:
        self.path = path
        self.element = element
        self.condition = condition

    def __str__(self) -> str:
        path = str(self.path) + str(self.element)
        if self.condition:
            path += "[%s]" % self.condition
        return path

    def __repr__(self) -> str:
        return "%s[%s]" % (self.__class__.__name__, self)

    def add_condition(self, condition: str, conjuction: str = "and") -> "XPathExpr":
        if self.condition:
            self.condition = "(%s) %s (%s)" % (self.condition, conjuction, condition)
        else:
            self.condition = condition
        return self

    def add_name_test(self) -> None:
        if self.element == "*":
            # We weren't doing a test anyway
            return
        self.add_condition("name() = %s" % GenericTranslator.xpath_literal(self.element))
        self.element = "*"

    def add_star_prefix(self) -> None:
        """
        Append '*/' to the path to keep the context constrained
        to a single parent.
        """
        self.path += "*/"

    def join(
        self,
        combiner: str,
        other: "XPathExpr",
        closing_combiner: Optional[str] = None,
        has_inner_condition: bool = False,
    ) -> "XPathExpr":
        path = str(self) + combiner
        # Any "star prefix" is redundant when joining.
        if other.path != "*/":
            path += other.path
        self.path = path
        if not has_inner_condition:
            self.element = other.element + closing_combiner if closing_combiner else other.element
            self.condition = other.condition
        else:
            self.element = other.element
            if other.condition:
                self.element += "[" + other.condition + "]"
            if closing_combiner:
                self.element += closing_combiner
        return self


split_at_single_quotes = re.compile("('+)").split

# The spec is actually more permissive than that, but don’t bother.
# This is just for the fast path.
# http://www.w3.org/TR/REC-xml/#NT-NameStartChar
is_safe_name = re.compile("^[a-zA-Z_][a-zA-Z0-9_.-]*$").match

# Test that the string is not empty and does not contain whitespace
is_non_whitespace = re.compile(r"^[^ \t\r\n\f]+$").match


#### Translation


class GenericTranslator:
    """
    Translator for "generic" XML documents.

    Everything is case-sensitive, no assumption is made on the meaning
    of element names and attribute names.

    """

    ####
    ####  HERE BE DRAGONS
    ####
    ####  You are welcome to hook into this to change some behavior,
    ####  but do so at your own risks.
    ####  Until it has received a lot more work and review,
    ####  I reserve the right to change this API in backward-incompatible ways
    ####  with any minor version of cssselect.
    ####  See https://github.com/scrapy/cssselect/pull/22
    ####  -- Simon Sapin.
    ####

    combinator_mapping = {
        " ": "descendant",
        ">": "child",
        "+": "direct_adjacent",
        "~": "indirect_adjacent",
    }

    attribute_operator_mapping = {
        "exists": "exists",
        "=": "equals",
        "~=": "includes",
        "|=": "dashmatch",
        "^=": "prefixmatch",
        "$=": "suffixmatch",
        "*=": "substringmatch",
        "!=": "different",  # XXX Not in Level 3 but meh
    }

    #: The attribute used for ID selectors depends on the document language:
    #: http://www.w3.org/TR/selectors/#id-selectors
    id_attribute = "id"

    #: The attribute used for ``:lang()`` depends on the document language:
    #: http://www.w3.org/TR/selectors/#lang-pseudo
    lang_attribute = "xml:lang"

    #: The case sensitivity of document language element names,
    #: attribute names, and attribute values in selectors depends
    #: on the document language.
    #: http://www.w3.org/TR/selectors/#casesens
    #:
    #: When a document language defines one of these as case-insensitive,
    #: cssselect assumes that the document parser makes the parsed values
    #: lower-case. Making the selector lower-case too makes the comparaison
    #: case-insensitive.
    #:
    #: In HTML, element names and attributes names (but not attribute values)
    #: are case-insensitive. All of lxml.html, html5lib, BeautifulSoup4
    #: and HTMLParser make them lower-case in their parse result, so
    #: the assumption holds.
    lower_case_element_names = False
    lower_case_attribute_names = False
    lower_case_attribute_values = False

    # class used to represent and xpath expression
    xpathexpr_cls = XPathExpr

    def css_to_xpath(self, css: str, prefix: str = "descendant-or-self::") -> str:
        """Translate a *group of selectors* to XPath.

        Pseudo-elements are not supported here since XPath only knows
        about "real" elements.

        :param css:
            A *group of selectors* as a string.
        :param prefix:
            This string is prepended to the XPath expression for each selector.
            The default makes selectors scoped to the context node’s subtree.
        :raises:
            :class:`~cssselect.SelectorSyntaxError` on invalid selectors,
            :class:`ExpressionError` on unknown/unsupported selectors,
            including pseudo-elements.
        :returns:
            The equivalent XPath 1.0 expression as a string.

        """
        return " | ".join(
            self.selector_to_xpath(selector, prefix, translate_pseudo_elements=True)
            for selector in parse(css)
        )

    def selector_to_xpath(
        self,
        selector: Selector,
        prefix: str = "descendant-or-self::",
        translate_pseudo_elements: bool = False,
    ) -> str:
        """Translate a parsed selector to XPath.


        :param selector:
            A parsed :class:`Selector` object.
        :param prefix:
            This string is prepended to the resulting XPath expression.
            The default makes selectors scoped to the context node’s subtree.
        :param translate_pseudo_elements:
            Unless this is set to ``True`` (as :meth:`css_to_xpath` does),
            the :attr:`~Selector.pseudo_element` attribute of the selector
            is ignored.
            It is the caller's responsibility to reject selectors
            with pseudo-elements, or to account for them somehow.
        :raises:
            :class:`ExpressionError` on unknown/unsupported selectors.
        :returns:
            The equivalent XPath 1.0 expression as a string.

        """
        tree = getattr(selector, "parsed_tree", None)
        if not tree:
            raise TypeError("Expected a parsed selector, got %r" % (selector,))
        xpath = self.xpath(tree)
        assert isinstance(xpath, self.xpathexpr_cls)  # help debug a missing 'return'
        if translate_pseudo_elements and selector.pseudo_element:
            xpath = self.xpath_pseudo_element(xpath, selector.pseudo_element)
        return (prefix or "") + str(xpath)

    def xpath_pseudo_element(self, xpath: XPathExpr, pseudo_element: PseudoElement) -> XPathExpr:
        """Translate a pseudo-element.

        Defaults to not supporting pseudo-elements at all,
        but can be overridden by sub-classes.

        """
        raise ExpressionError("Pseudo-elements are not supported.")

    @staticmethod
    def xpath_literal(s: str) -> str:
        s = str(s)
        if "'" not in s:
            s = "'%s'" % s
        elif '"' not in s:
            s = '"%s"' % s
        else:
            s = "concat(%s)" % ",".join(
                [
                    (("'" in part) and '"%s"' or "'%s'") % part
                    for part in split_at_single_quotes(s)
                    if part
                ]
            )
        return s

    def xpath(self, parsed_selector: Tree) -> XPathExpr:
        """Translate any parsed selector object."""
        type_name = type(parsed_selector).__name__
        method = getattr(self, "xpath_%s" % type_name.lower(), None)
        if method is None:
            raise ExpressionError("%s is not supported." % type_name)
        return typing.cast(XPathExpr, method(parsed_selector))

    # Dispatched by parsed object type

    def xpath_combinedselector(self, combined: CombinedSelector) -> XPathExpr:
        """Translate a combined selector."""
        combinator = self.combinator_mapping[combined.combinator]
        method = getattr(self, "xpath_%s_combinator" % combinator)
        return typing.cast(
            XPathExpr, method(self.xpath(combined.selector), self.xpath(combined.subselector))
        )

    def xpath_negation(self, negation: Negation) -> XPathExpr:
        xpath = self.xpath(negation.selector)
        sub_xpath = self.xpath(negation.subselector)
        sub_xpath.add_name_test()
        if sub_xpath.condition:
            return xpath.add_condition("not(%s)" % sub_xpath.condition)
        else:
            return xpath.add_condition("0")

    def xpath_relation(self, relation: Relation) -> XPathExpr:
        xpath = self.xpath(relation.selector)
        combinator = relation.combinator
        subselector = relation.subselector
        right = self.xpath(subselector.parsed_tree)
        method = getattr(
            self,
            "xpath_relation_%s_combinator"
            % self.combinator_mapping[typing.cast(str, combinator.value)],
        )
        return typing.cast(XPathExpr, method(xpath, right))

    def xpath_matching(self, matching: Matching) -> XPathExpr:
        xpath = self.xpath(matching.selector)
        exprs = [self.xpath(selector) for selector in matching.selector_list]
        for e in exprs:
            e.add_name_test()
            if e.condition:
                xpath.add_condition(e.condition, "or")
        return xpath

    def xpath_specificityadjustment(self, matching: SpecificityAdjustment) -> XPathExpr:
        xpath = self.xpath(matching.selector)
        exprs = [self.xpath(selector) for selector in matching.selector_list]
        for e in exprs:
            e.add_name_test()
            if e.condition:
                xpath.add_condition(e.condition, "or")
        return xpath

    def xpath_function(self, function: Function) -> XPathExpr:
        """Translate a functional pseudo-class."""
        method_name = "xpath_%s_function" % function.name.replace("-", "_")
        method = getattr(self, method_name, None)
        if not method:
            raise ExpressionError("The pseudo-class :%s() is unknown" % function.name)
        return typing.cast(XPathExpr, method(self.xpath(function.selector), function))

    def xpath_pseudo(self, pseudo: Pseudo) -> XPathExpr:
        """Translate a pseudo-class."""
        method_name = "xpath_%s_pseudo" % pseudo.ident.replace("-", "_")
        method = getattr(self, method_name, None)
        if not method:
            # TODO: better error message for pseudo-elements?
            raise ExpressionError("The pseudo-class :%s is unknown" % pseudo.ident)
        return typing.cast(XPathExpr, method(self.xpath(pseudo.selector)))

    def xpath_attrib(self, selector: Attrib) -> XPathExpr:
        """Translate an attribute selector."""
        operator = self.attribute_operator_mapping[selector.operator]
        method = getattr(self, "xpath_attrib_%s" % operator)
        if self.lower_case_attribute_names:
            name = selector.attrib.lower()
        else:
            name = selector.attrib
        safe = is_safe_name(name)
        if selector.namespace:
            name = "%s:%s" % (selector.namespace, name)
            safe = safe and is_safe_name(selector.namespace)
        if safe:
            attrib = "@" + name
        else:
            attrib = "attribute::*[name() = %s]" % self.xpath_literal(name)
        if selector.value is None:
            value = None
        elif self.lower_case_attribute_values:
            value = typing.cast(str, selector.value.value).lower()
        else:
            value = selector.value.value
        return typing.cast(XPathExpr, method(self.xpath(selector.selector), attrib, value))

    def xpath_class(self, class_selector: Class) -> XPathExpr:
        """Translate a class selector."""
        # .foo is defined as [class~=foo] in the spec.
        xpath = self.xpath(class_selector.selector)
        return self.xpath_attrib_includes(xpath, "@class", class_selector.class_name)

    def xpath_hash(self, id_selector: Hash) -> XPathExpr:
        """Translate an ID selector."""
        xpath = self.xpath(id_selector.selector)
        return self.xpath_attrib_equals(xpath, "@id", id_selector.id)

    def xpath_element(self, selector: Element) -> XPathExpr:
        """Translate a type or universal selector."""
        element = selector.element
        if not element:
            element = "*"
            safe = True
        else:
            safe = bool(is_safe_name(element))
            if self.lower_case_element_names:
                element = element.lower()
        if selector.namespace:
            # Namespace prefixes are case-sensitive.
            # http://www.w3.org/TR/css3-namespace/#prefixes
            element = "%s:%s" % (selector.namespace, element)
            safe = safe and bool(is_safe_name(selector.namespace))
        xpath = self.xpathexpr_cls(element=element)
        if not safe:
            xpath.add_name_test()
        return xpath

    # CombinedSelector: dispatch by combinator

    def xpath_descendant_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a child, grand-child or further descendant of left"""
        return left.join("/descendant-or-self::*/", right)

    def xpath_child_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is an immediate child of left"""
        return left.join("/", right)

    def xpath_direct_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a sibling immediately after left"""
        xpath = left.join("/following-sibling::", right)
        xpath.add_name_test()
        return xpath.add_condition("position() = 1")

    def xpath_indirect_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a sibling after left, immediately or not"""
        return left.join("/following-sibling::", right)

    def xpath_relation_descendant_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a child, grand-child or further descendant of left; select left"""
        return left.join("[descendant::", right, closing_combiner="]", has_inner_condition=True)

    def xpath_relation_child_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is an immediate child of left; select left"""
        return left.join("[./", right, closing_combiner="]")

    def xpath_relation_direct_adjacent_combinator(
        self, left: XPathExpr, right: XPathExpr
    ) -> XPathExpr:
        """right is a sibling immediately after left; select left"""
        xpath = left.add_condition(
            "following-sibling::*[(name() = '{}') and (position() = 1)]".format(right.element)
        )
        return xpath

    def xpath_relation_indirect_adjacent_combinator(
        self, left: XPathExpr, right: XPathExpr
    ) -> XPathExpr:
        """right is a sibling after left, immediately or not; select left"""
        return left.join("[following-sibling::", right, closing_combiner="]")

    # Function: dispatch by function/pseudo-class name

    def xpath_nth_child_function(
        self, xpath: XPathExpr, function: Function, last: bool = False, add_name_test: bool = True
    ) -> XPathExpr:
        try:
            a, b = parse_series(function.arguments)
        except ValueError:
            raise ExpressionError("Invalid series: '%r'" % function.arguments)

        # From https://www.w3.org/TR/css3-selectors/#structural-pseudos:
        #
        # :nth-child(an+b)
        #       an+b-1 siblings before
        #
        # :nth-last-child(an+b)
        #       an+b-1 siblings after
        #
        # :nth-of-type(an+b)
        #       an+b-1 siblings with the same expanded element name before
        #
        # :nth-last-of-type(an+b)
        #       an+b-1 siblings with the same expanded element name after
        #
        # So,
        # for :nth-child and :nth-of-type
        #
        #    count(preceding-sibling::<nodetest>) = an+b-1
        #
        # for :nth-last-child and :nth-last-of-type
        #
        #    count(following-sibling::<nodetest>) = an+b-1
        #
        # therefore,
        #    count(...) - (b-1) ≡ 0 (mod a)
        #
        # if a == 0:
        # ~~~~~~~~~~
        #    count(...) = b-1
        #
        # if a < 0:
        # ~~~~~~~~~
        #    count(...) - b +1 <= 0
        # -> count(...) <= b-1
        #
        # if a > 0:
        # ~~~~~~~~~
        #    count(...) - b +1 >= 0
        # -> count(...) >= b-1

        # work with b-1 instead
        b_min_1 = b - 1

        # early-exit condition 1:
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # for a == 1, nth-*(an+b) means n+b-1 siblings before/after,
        # and since n ∈ {0, 1, 2, ...}, if b-1<=0,
        # there is always an "n" matching any number of siblings (maybe none)
        if a == 1 and b_min_1 <= 0:
            return xpath

        # early-exit condition 2:
        # ~~~~~~~~~~~~~~~~~~~~~~~
        # an+b-1 siblings with a<0 and (b-1)<0 is not possible
        if a < 0 and b_min_1 < 0:
            return xpath.add_condition("0")

        # `add_name_test` boolean is inverted and somewhat counter-intuitive:
        #
        # nth_of_type() calls nth_child(add_name_test=False)
        if add_name_test:
            nodetest = "*"
        else:
            nodetest = "%s" % xpath.element

        # count siblings before or after the element
        if not last:
            siblings_count = "count(preceding-sibling::%s)" % nodetest
        else:
            siblings_count = "count(following-sibling::%s)" % nodetest

        # special case of fixed position: nth-*(0n+b)
        # if a == 0:
        # ~~~~~~~~~~
        #    count(***-sibling::***) = b-1
        if a == 0:
            return xpath.add_condition("%s = %s" % (siblings_count, b_min_1))

        expressions = []

        if a > 0:
            # siblings count, an+b-1, is always >= 0,
            # so if a>0, and (b-1)<=0, an "n" exists to satisfy this,
            # therefore, the predicate is only interesting if (b-1)>0
            if b_min_1 > 0:
                expressions.append("%s >= %s" % (siblings_count, b_min_1))
        else:
            # if a<0, and (b-1)<0, no "n" satisfies this,
            # this is tested above as an early exist condition
            # otherwise,
            expressions.append("%s <= %s" % (siblings_count, b_min_1))

        # operations modulo 1 or -1 are simpler, one only needs to verify:
        #
        # - either:
        # count(***-sibling::***) - (b-1) = n = 0, 1, 2, 3, etc.,
        #   i.e. count(***-sibling::***) >= (b-1)
        #
        # - or:
        # count(***-sibling::***) - (b-1) = -n = 0, -1, -2, -3, etc.,
        #   i.e. count(***-sibling::***) <= (b-1)
        # we we just did above.
        #
        if abs(a) != 1:
            # count(***-sibling::***) - (b-1) ≡ 0 (mod a)
            left = siblings_count

            # apply "modulo a" on 2nd term, -(b-1),
            # to simplify things like "(... +6) % -3",
            # and also make it positive with |a|
            b_neg = (-b_min_1) % abs(a)

            if b_neg != 0:
                b_neg_as_str = "+%s" % b_neg
                left = "(%s %s)" % (left, b_neg_as_str)

            expressions.append("%s mod %s = 0" % (left, a))

        if len(expressions) > 1:
            template = "(%s)"
        else:
            template = "%s"
        xpath.add_condition(" and ".join(template % expression for expression in expressions))
        return xpath

    def xpath_nth_last_child_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        return self.xpath_nth_child_function(xpath, function, last=True)

    def xpath_nth_of_type_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        if xpath.element == "*":
            raise ExpressionError("*:nth-of-type() is not implemented")
        return self.xpath_nth_child_function(xpath, function, add_name_test=False)

    def xpath_nth_last_of_type_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        if xpath.element == "*":
            raise ExpressionError("*:nth-of-type() is not implemented")
        return self.xpath_nth_child_function(xpath, function, last=True, add_name_test=False)

    def xpath_contains_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        # Defined there, removed in later drafts:
        # http://www.w3.org/TR/2001/CR-css3-selectors-20011113/#content-selectors
        if function.argument_types() not in (["STRING"], ["IDENT"]):
            raise ExpressionError(
                "Expected a single string or ident for :contains(), got %r" % function.arguments
            )
        value = typing.cast(str, function.arguments[0].value)
        return xpath.add_condition("contains(., %s)" % self.xpath_literal(value))

    def xpath_lang_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        if function.argument_types() not in (["STRING"], ["IDENT"]):
            raise ExpressionError(
                "Expected a single string or ident for :lang(), got %r" % function.arguments
            )
        value = typing.cast(str, function.arguments[0].value)
        return xpath.add_condition("lang(%s)" % (self.xpath_literal(value)))

    # Pseudo: dispatch by pseudo-class name

    def xpath_root_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition("not(parent::*)")

    # CSS immediate children (CSS ":scope > div" to XPath "child::div" or "./div")
    # Works only at the start of a selector
    # Needed to get immediate children of a processed selector in Scrapy
    # for product in response.css('.product'):
    #     description = product.css(':scope > div::text').get()
    def xpath_scope_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition("1")

    def xpath_first_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition("count(preceding-sibling::*) = 0")

    def xpath_last_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition("count(following-sibling::*) = 0")

    def xpath_first_of_type_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        if xpath.element == "*":
            raise ExpressionError("*:first-of-type is not implemented")
        return xpath.add_condition("count(preceding-sibling::%s) = 0" % xpath.element)

    def xpath_last_of_type_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        if xpath.element == "*":
            raise ExpressionError("*:last-of-type is not implemented")
        return xpath.add_condition("count(following-sibling::%s) = 0" % xpath.element)

    def xpath_only_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition("count(parent::*/child::*) = 1")

    def xpath_only_of_type_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        if xpath.element == "*":
            raise ExpressionError("*:only-of-type is not implemented")
        return xpath.add_condition("count(parent::*/child::%s) = 1" % xpath.element)

    def xpath_empty_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition("not(*) and not(string-length())")

    def pseudo_never_matches(self, xpath: XPathExpr) -> XPathExpr:
        """Common implementation for pseudo-classes that never match."""
        return xpath.add_condition("0")

    xpath_link_pseudo = pseudo_never_matches
    xpath_visited_pseudo = pseudo_never_matches
    xpath_hover_pseudo = pseudo_never_matches
    xpath_active_pseudo = pseudo_never_matches
    xpath_focus_pseudo = pseudo_never_matches
    xpath_target_pseudo = pseudo_never_matches
    xpath_enabled_pseudo = pseudo_never_matches
    xpath_disabled_pseudo = pseudo_never_matches
    xpath_checked_pseudo = pseudo_never_matches

    # Attrib: dispatch by attribute operator

    def xpath_attrib_exists(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        assert not value
        xpath.add_condition(name)
        return xpath

    def xpath_attrib_equals(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        assert value is not None
        xpath.add_condition("%s = %s" % (name, self.xpath_literal(value)))
        return xpath

    def xpath_attrib_different(
        self, xpath: XPathExpr, name: str, value: Optional[str]
    ) -> XPathExpr:
        assert value is not None
        # FIXME: this seems like a weird hack...
        if value:
            xpath.add_condition("not(%s) or %s != %s" % (name, name, self.xpath_literal(value)))
        else:
            xpath.add_condition("%s != %s" % (name, self.xpath_literal(value)))
        return xpath

    def xpath_attrib_includes(
        self, xpath: XPathExpr, name: str, value: Optional[str]
    ) -> XPathExpr:
        if value and is_non_whitespace(value):
            xpath.add_condition(
                "%s and contains(concat(' ', normalize-space(%s), ' '), %s)"
                % (name, name, self.xpath_literal(" " + value + " "))
            )
        else:
            xpath.add_condition("0")
        return xpath

    def xpath_attrib_dashmatch(
        self, xpath: XPathExpr, name: str, value: Optional[str]
    ) -> XPathExpr:
        assert value is not None
        # Weird, but true...
        xpath.add_condition(
            "%s and (%s = %s or starts-with(%s, %s))"
            % (name, name, self.xpath_literal(value), name, self.xpath_literal(value + "-"))
        )
        return xpath

    def xpath_attrib_prefixmatch(
        self, xpath: XPathExpr, name: str, value: Optional[str]
    ) -> XPathExpr:
        if value:
            xpath.add_condition(
                "%s and starts-with(%s, %s)" % (name, name, self.xpath_literal(value))
            )
        else:
            xpath.add_condition("0")
        return xpath

    def xpath_attrib_suffixmatch(
        self, xpath: XPathExpr, name: str, value: Optional[str]
    ) -> XPathExpr:
        if value:
            # Oddly there is a starts-with in XPath 1.0, but not ends-with
            xpath.add_condition(
                "%s and substring(%s, string-length(%s)-%s) = %s"
                % (name, name, name, len(value) - 1, self.xpath_literal(value))
            )
        else:
            xpath.add_condition("0")
        return xpath

    def xpath_attrib_substringmatch(
        self, xpath: XPathExpr, name: str, value: Optional[str]
    ) -> XPathExpr:
        if value:
            # Attribute selectors are case sensitive
            xpath.add_condition(
                "%s and contains(%s, %s)" % (name, name, self.xpath_literal(value))
            )
        else:
            xpath.add_condition("0")
        return xpath


class HTMLTranslator(GenericTranslator):
    """
    Translator for (X)HTML documents.

    Has a more useful implementation of some pseudo-classes based on
    HTML-specific element names and attribute names, as described in
    the `HTML5 specification`_. It assumes no-quirks mode.
    The API is the same as :class:`GenericTranslator`.

    .. _HTML5 specification: http://www.w3.org/TR/html5/links.html#selectors

    :param xhtml:
        If false (the default), element names and attribute names
        are case-insensitive.

    """

    lang_attribute = "lang"

    def __init__(self, xhtml: bool = False) -> None:
        self.xhtml = xhtml  # Might be useful for sub-classes?
        if not xhtml:
            # See their definition in GenericTranslator.
            self.lower_case_element_names = True
            self.lower_case_attribute_names = True

    def xpath_checked_pseudo(self, xpath: XPathExpr) -> XPathExpr:  # type: ignore
        # FIXME: is this really all the elements?
        return xpath.add_condition(
            "(@selected and name(.) = 'option') or "
            "(@checked "
            "and (name(.) = 'input' or name(.) = 'command')"
            "and (@type = 'checkbox' or @type = 'radio'))"
        )

    def xpath_lang_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        if function.argument_types() not in (["STRING"], ["IDENT"]):
            raise ExpressionError(
                "Expected a single string or ident for :lang(), got %r" % function.arguments
            )
        value = function.arguments[0].value
        assert value
        return xpath.add_condition(
            "ancestor-or-self::*[@lang][1][starts-with(concat("
            # XPath 1.0 has no lower-case function...
            "translate(@%s, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz'), "
            "'-'), %s)]" % (self.lang_attribute, self.xpath_literal(value.lower() + "-"))
        )

    def xpath_link_pseudo(self, xpath: XPathExpr) -> XPathExpr:  # type: ignore
        return xpath.add_condition(
            "@href and (name(.) = 'a' or name(.) = 'link' or name(.) = 'area')"
        )

    # Links are never visited, the implementation for :visited is the same
    # as in GenericTranslator

    def xpath_disabled_pseudo(self, xpath: XPathExpr) -> XPathExpr:  # type: ignore
        # http://www.w3.org/TR/html5/section-index.html#attributes-1
        return xpath.add_condition(
            """
        (
            @disabled and
            (
                (name(.) = 'input' and @type != 'hidden') or
                name(.) = 'button' or
                name(.) = 'select' or
                name(.) = 'textarea' or
                name(.) = 'command' or
                name(.) = 'fieldset' or
                name(.) = 'optgroup' or
                name(.) = 'option'
            )
        ) or (
            (
                (name(.) = 'input' and @type != 'hidden') or
                name(.) = 'button' or
                name(.) = 'select' or
                name(.) = 'textarea'
            )
            and ancestor::fieldset[@disabled]
        )
        """
        )
        # FIXME: in the second half, add "and is not a descendant of that
        # fieldset element's first legend element child, if any."

    def xpath_enabled_pseudo(self, xpath: XPathExpr) -> XPathExpr:  # type: ignore
        # http://www.w3.org/TR/html5/section-index.html#attributes-1
        return xpath.add_condition(
            """
        (
            @href and (
                name(.) = 'a' or
                name(.) = 'link' or
                name(.) = 'area'
            )
        ) or (
            (
                name(.) = 'command' or
                name(.) = 'fieldset' or
                name(.) = 'optgroup'
            )
            and not(@disabled)
        ) or (
            (
                (name(.) = 'input' and @type != 'hidden') or
                name(.) = 'button' or
                name(.) = 'select' or
                name(.) = 'textarea' or
                name(.) = 'keygen'
            )
            and not (@disabled or ancestor::fieldset[@disabled])
        ) or (
            name(.) = 'option' and not(
                @disabled or ancestor::optgroup[@disabled]
            )
        )
        """
        )
        # FIXME: ... or "li elements that are children of menu elements,
        # and that have a child element that defines a command, if the first
        # such element's Disabled State facet is false (not disabled)".
        # FIXME: after ancestor::fieldset[@disabled], add "and is not a
        # descendant of that fieldset element's first legend element child,
        # if any."
