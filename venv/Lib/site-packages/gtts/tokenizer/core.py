# -*- coding: utf-8 -*-
import re


class RegexBuilder:
    r"""Builds regex using arguments passed into a pattern template.

    Builds a regex object for which the pattern is made from an argument
    passed into a template. If more than one argument is passed (iterable),
    each pattern is joined by "|" (regex alternation 'or') to create a
    single pattern.

    Args:
        pattern_args (iteratable): String element(s) to be each passed to
            ``pattern_func`` to create a regex pattern. Each element is
            ``re.escape``'d before being passed.
        pattern_func (callable): A 'template' function that should take a
            string and return a string. It should take an element of
            ``pattern_args`` and return a valid regex pattern group string.
        flags: ``re`` flag(s) to compile with the regex.

    Example:
        To create a simple regex that matches on the characters "a", "b",
        or "c", followed by a period::

            >>> rb = RegexBuilder('abc', lambda x: "{}\.".format(x))

        Looking at ``rb.regex`` we get the following compiled regex::

            >>> print(rb.regex)
            'a\.|b\.|c\.'

        The above is fairly simple, but this class can help in writing more
        complex repetitive regex, making them more readable and easier to
        create by using existing data structures.

    Example:
        To match the character following the words "lorem", "ipsum", "meili"
        or "koda"::

            >>> words = ['lorem', 'ipsum', 'meili', 'koda']
            >>> rb = RegexBuilder(words, lambda x: "(?<={}).".format(x))

        Looking at ``rb.regex`` we get the following compiled regex::

            >>> print(rb.regex)
            '(?<=lorem).|(?<=ipsum).|(?<=meili).|(?<=koda).'

    """

    def __init__(self, pattern_args, pattern_func, flags=0):
        self.pattern_args = pattern_args
        self.pattern_func = pattern_func
        self.flags = flags

        # Compile
        self.regex = self._compile()

    def _compile(self):
        alts = []
        for arg in self.pattern_args:
            arg = re.escape(arg)
            alt = self.pattern_func(arg)
            alts.append(alt)

        pattern = "|".join(alts)
        return re.compile(pattern, self.flags)

    def __repr__(self):  # pragma: no cover
        return str(self.regex)


class PreProcessorRegex:
    r"""Regex-based substitution text pre-processor.

    Runs a series of regex substitutions (``re.sub``) from each ``regex`` of a
    :class:`gtts.tokenizer.core.RegexBuilder` with an extra ``repl``
    replacement parameter.

    Args:
        search_args (iteratable): String element(s) to be each passed to
            ``search_func`` to create a regex pattern. Each element is
            ``re.escape``'d before being passed.
        search_func (callable): A 'template' function that should take a
            string and return a string. It should take an element of
            ``search_args`` and return a valid regex search pattern string.
        repl (string): The common replacement passed to the ``sub`` method for
            each ``regex``. Can be a raw string (the case of a regex
            backreference, for example)
        flags: ``re`` flag(s) to compile with each `regex`.

    Example:
        Add "!" after the words "lorem" or "ipsum", while ignoring case::

            >>> import re
            >>> words = ['lorem', 'ipsum']
            >>> pp = PreProcessorRegex(words,
            ...                        lambda x: "({})".format(x), r'\\1!',
            ...                        re.IGNORECASE)

        In this case, the regex is a group and the replacement uses its
        backreference ``\\1`` (as a raw string). Looking at ``pp`` we get the
        following list of search/replacement pairs::

            >>> print(pp)
            (re.compile('(lorem)', re.IGNORECASE), repl='\1!'),
            (re.compile('(ipsum)', re.IGNORECASE), repl='\1!')

        It can then be run on any string of text::

            >>> pp.run("LOREM ipSuM")
            "LOREM! ipSuM!"

    See :mod:`gtts.tokenizer.pre_processors` for more examples.

    """

    def __init__(self, search_args, search_func, repl, flags=0):
        self.repl = repl

        # Create regex list
        self.regexes = []
        for arg in search_args:
            rb = RegexBuilder([arg], search_func, flags)
            self.regexes.append(rb.regex)

    def run(self, text):
        """Run each regex substitution on ``text``.

        Args:
            text (string): the input text.

        Returns:
            string: text after all substitutions have been sequentially
            applied.

        """
        for regex in self.regexes:
            text = regex.sub(self.repl, text)
        return text

    def __repr__(self):  # pragma: no cover
        subs_strs = []
        for r in self.regexes:
            subs_strs.append("({}, repl='{}')".format(r, self.repl))
        return ", ".join(subs_strs)


class PreProcessorSub:
    r"""Simple substitution text preprocessor.

    Performs string-for-string substitution from list a find/replace pairs.
    It abstracts :class:`gtts.tokenizer.core.PreProcessorRegex` with a default
    simple substitution regex.

    Args:
        sub_pairs (list): A list of tuples of the style
            ``(<search str>, <replace str>)``
        ignore_case (bool): Ignore case during search. Defaults to ``True``.

    Example:
        Replace all occurences of "Mac" to "PC" and "Firefox" to "Chrome"::

            >>> sub_pairs = [('Mac', 'PC'), ('Firefox', 'Chrome')]
            >>> pp = PreProcessorSub(sub_pairs)

        Looking at the ``pp``, we get the following list of
        search (regex)/replacement pairs::

            >>> print(pp)
            (re.compile('Mac', re.IGNORECASE), repl='PC'),
            (re.compile('Firefox', re.IGNORECASE), repl='Chrome')

        It can then be run on any string of text::

            >>> pp.run("I use firefox on my mac")
            "I use Chrome on my PC"

    See :mod:`gtts.tokenizer.pre_processors` for more examples.

    """

    def __init__(self, sub_pairs, ignore_case=True):
        def search_func(x):
            return u"{}".format(x)

        flags = re.I if ignore_case else 0

        # Create pre-processor list
        self.pre_processors = []
        for sub_pair in sub_pairs:
            pattern, repl = sub_pair
            pp = PreProcessorRegex([pattern], search_func, repl, flags)
            self.pre_processors.append(pp)

    def run(self, text):
        """Run each substitution on ``text``.

        Args:
            text (string): the input text.

        Returns:
            string: text after all substitutions have been sequentially
            applied.

        """
        for pp in self.pre_processors:
            text = pp.run(text)
        return text

    def __repr__(self):  # pragma: no cover
        return ", ".join([str(pp) for pp in self.pre_processors])


class Tokenizer:
    r"""An extensible but simple generic rule-based tokenizer.

    A generic and simple string tokenizer that takes a list of functions
    (called `tokenizer cases`) returning ``regex`` objects and joins them by
    "|" (regex alternation 'or') to create a single regex to use with the
    standard ``regex.split()`` function.

    ``regex_funcs`` is a list of any function that can return a ``regex``
    (from ``re.compile()``) object, such as a
    :class:`gtts.tokenizer.core.RegexBuilder` instance (and its ``regex``
    attribute).

    See the :mod:`gtts.tokenizer.tokenizer_cases` module for examples.

    Args:
        regex_funcs (list): List of compiled ``regex`` objects. Each
            functions's pattern will be joined into a single pattern and
            compiled.
        flags: ``re`` flag(s) to compile with the final regex. Defaults to
            ``re.IGNORECASE``

    Note:
        When the ``regex`` objects obtained from ``regex_funcs`` are joined,
        their individual ``re`` flags are ignored in favour of ``flags``.

    Raises:
        TypeError: When an element of ``regex_funcs`` is not a function, or
            a function that does not return a compiled ``regex`` object.

    Warning:
        Joined ``regex`` patterns can easily interfere with one another in
        unexpected ways. It is recommanded that each tokenizer case operate
        on distinct or non-overlapping chracters/sets of characters
        (For example, a tokenizer case for the period (".") should also
        handle not matching/cutting on decimals, instead of making that
        a seperate tokenizer case).

    Example:
        A tokenizer with a two simple case (*Note: these are bad cases to
        tokenize on, this is simply a usage example*)::

            >>> import re, RegexBuilder
            >>>
            >>> def case1():
            ...     return re.compile("\,")
            >>>
            >>> def case2():
            ...     return RegexBuilder('abc', lambda x: "{}\.".format(x)).regex
            >>>
            >>> t = Tokenizer([case1, case2])

        Looking at ``case1().pattern``, we get::

            >>> print(case1().pattern)
            '\\,'

        Looking at ``case2().pattern``, we get::

            >>> print(case2().pattern)
            'a\\.|b\\.|c\\.'

        Finally, looking at ``t``, we get them combined::

            >>> print(t)
            're.compile('\\,|a\\.|b\\.|c\\.', re.IGNORECASE)
             from: [<function case1 at 0x10bbcdd08>, <function case2 at 0x10b5c5e18>]'

        It can then be run on any string of text::

            >>> t.run("Hello, my name is Linda a. Call me Lin, b. I'm your friend")
            ['Hello', ' my name is Linda ', ' Call me Lin', ' ', " I'm your friend"]

    """

    def __init__(self, regex_funcs, flags=re.IGNORECASE):
        self.regex_funcs = regex_funcs
        self.flags = flags

        try:
            # Combine
            self.total_regex = self._combine_regex()
        except (TypeError, AttributeError) as e:  # pragma: no cover
            raise TypeError(
                "Tokenizer() expects a list of functions returning "
                "regular expression objects (i.e. re.compile). " + str(e)
            )

    def _combine_regex(self):
        alts = []
        for func in self.regex_funcs:
            alts.append(func())

        pattern = "|".join(alt.pattern for alt in alts)
        return re.compile(pattern, self.flags)

    def run(self, text):
        """Tokenize `text`.

        Args:
            text (string): the input text to tokenize.

        Returns:
            list: A list of strings (token) split according to the tokenizer cases.

        """
        return self.total_regex.split(text)

    def __repr__(self):  # pragma: no cover
        return str(self.total_regex) + " from: " + str(self.regex_funcs)
