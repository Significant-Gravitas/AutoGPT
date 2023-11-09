# -*- coding: utf-8 -*-
"""
    CSS Selectors based on XPath
    ============================

    This module supports selecting XML/HTML elements based on CSS selectors.
    See the `CSSSelector` class for details.


    :copyright: (c) 2007-2012 Ian Bicking and contributors.
                See AUTHORS for more details.
    :license: BSD, see LICENSE for more details.

"""

from cssselect.parser import (
    parse,
    Selector,
    FunctionalPseudoElement,
    SelectorError,
    SelectorSyntaxError,
)
from cssselect.xpath import GenericTranslator, HTMLTranslator, ExpressionError

__all__ = (
    "ExpressionError",
    "FunctionalPseudoElement",
    "GenericTranslator",
    "HTMLTranslator",
    "parse",
    "Selector",
    "SelectorError",
    "SelectorSyntaxError",
)

VERSION = "1.2.0"
__version__ = VERSION
