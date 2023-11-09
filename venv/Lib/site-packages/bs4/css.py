"""Integration code for CSS selectors using Soup Sieve (pypi: soupsieve)."""

import warnings
try:
    import soupsieve
except ImportError as e:
    soupsieve = None
    warnings.warn(
        'The soupsieve package is not installed. CSS selectors cannot be used.'
    )


class CSS(object):
    """A proxy object against the soupsieve library, to simplify its
    CSS selector API.

    Acquire this object through the .css attribute on the
    BeautifulSoup object, or on the Tag you want to use as the
    starting point for a CSS selector.

    The main advantage of doing this is that the tag to be selected
    against doesn't need to be explicitly specified in the function
    calls, since it's already scoped to a tag.
    """

    def __init__(self, tag, api=soupsieve):
        """Constructor.

        You don't need to instantiate this class yourself; instead,
        access the .css attribute on the BeautifulSoup object, or on
        the Tag you want to use as the starting point for your CSS
        selector.

        :param tag: All CSS selectors will use this as their starting
        point.

        :param api: A plug-in replacement for the soupsieve module,
        designed mainly for use in tests.
        """
        if api is None:
            raise NotImplementedError(
                "Cannot execute CSS selectors because the soupsieve package is not installed."
            )
        self.api = api
        self.tag = tag

    def escape(self, ident):
        """Escape a CSS identifier.

        This is a simple wrapper around soupselect.escape(). See the
        documentation for that function for more information.
        """
        if soupsieve is None:
            raise NotImplementedError(
                "Cannot escape CSS identifiers because the soupsieve package is not installed."
            )
        return self.api.escape(ident)

    def _ns(self, ns, select):
        """Normalize a dictionary of namespaces."""
        if not isinstance(select, self.api.SoupSieve) and ns is None:
            # If the selector is a precompiled pattern, it already has
            # a namespace context compiled in, which cannot be
            # replaced.
            ns = self.tag._namespaces
        return ns

    def _rs(self, results):
        """Normalize a list of results to a Resultset.

        A ResultSet is more consistent with the rest of Beautiful
        Soup's API, and ResultSet.__getattr__ has a helpful error
        message if you try to treat a list of results as a single
        result (a common mistake).
        """
        # Import here to avoid circular import
        from bs4.element import ResultSet
        return ResultSet(None, results)

    def compile(self, select, namespaces=None, flags=0, **kwargs):
        """Pre-compile a selector and return the compiled object.

        :param selector: A CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
           used in the CSS selector to namespace URIs. By default,
           Beautiful Soup will use the prefixes it encountered while
           parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            soupsieve.compile() method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
           soupsieve.compile() method.

        :return: A precompiled selector object.
        :rtype: soupsieve.SoupSieve
        """
        return self.api.compile(
            select, self._ns(namespaces, select), flags, **kwargs
        )

    def select_one(self, select, namespaces=None, flags=0, **kwargs):
        """Perform a CSS selection operation on the current Tag and return the
        first result.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the soupsieve.select_one()
        method.

        :param selector: A CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
           used in the CSS selector to namespace URIs. By default,
           Beautiful Soup will use the prefixes it encountered while
           parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            soupsieve.select_one() method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
           soupsieve.select_one() method.

        :return: A Tag, or None if the selector has no match.
        :rtype: bs4.element.Tag

        """
        return self.api.select_one(
            select, self.tag, self._ns(namespaces, select), flags, **kwargs
        )

    def select(self, select, namespaces=None, limit=0, flags=0, **kwargs):
        """Perform a CSS selection operation on the current Tag.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the soupsieve.select()
        method.

        :param selector: A string containing a CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param limit: After finding this number of results, stop looking.

        :param flags: Flags to be passed into Soup Sieve's
            soupsieve.select() method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
            soupsieve.select() method.

        :return: A ResultSet of Tag objects.
        :rtype: bs4.element.ResultSet

        """
        if limit is None:
            limit = 0

        return self._rs(
            self.api.select(
                select, self.tag, self._ns(namespaces, select), limit, flags,
                **kwargs
            )
        )

    def iselect(self, select, namespaces=None, limit=0, flags=0, **kwargs):
        """Perform a CSS selection operation on the current Tag.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the soupsieve.iselect()
        method. It is the same as select(), but it returns a generator
        instead of a list.

        :param selector: A string containing a CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param limit: After finding this number of results, stop looking.

        :param flags: Flags to be passed into Soup Sieve's
            soupsieve.iselect() method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
            soupsieve.iselect() method.

        :return: A generator
        :rtype: types.GeneratorType
        """
        return self.api.iselect(
            select, self.tag, self._ns(namespaces, select), limit, flags, **kwargs
        )

    def closest(self, select, namespaces=None, flags=0, **kwargs):
        """Find the Tag closest to this one that matches the given selector.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the soupsieve.closest()
        method.

        :param selector: A string containing a CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            soupsieve.closest() method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
            soupsieve.closest() method.

        :return: A Tag, or None if there is no match.
        :rtype: bs4.Tag

        """
        return self.api.closest(
            select, self.tag, self._ns(namespaces, select), flags, **kwargs
        )

    def match(self, select, namespaces=None, flags=0, **kwargs):
        """Check whether this Tag matches the given CSS selector.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the soupsieve.match()
        method.

        :param: a CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            soupsieve.match() method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
            soupsieve.match() method.

        :return: True if this Tag matches the selector; False otherwise.
        :rtype: bool
        """
        return self.api.match(
            select, self.tag, self._ns(namespaces, select), flags, **kwargs
        )

    def filter(self, select, namespaces=None, flags=0, **kwargs):
        """Filter this Tag's direct children based on the given CSS selector.

        This uses the Soup Sieve library. It works the same way as
        passing this Tag into that library's soupsieve.filter()
        method. More information, for more information see the
        documentation for soupsieve.filter().

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            soupsieve.filter() method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
            soupsieve.filter() method.

        :return: A ResultSet of Tag objects.
        :rtype: bs4.element.ResultSet

        """
        return self._rs(
            self.api.filter(
                select, self.tag, self._ns(namespaces, select), flags, **kwargs
            )
        )
