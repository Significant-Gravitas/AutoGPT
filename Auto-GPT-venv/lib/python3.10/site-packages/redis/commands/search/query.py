class Query:
    """
    Query is used to build complex queries that have more parameters than just
    the query string. The query string is set in the constructor, and other
    options have setter functions.

    The setter functions return the query object, so they can be chained,
    i.e. `Query("foo").verbatim().filter(...)` etc.
    """

    def __init__(self, query_string):
        """
        Create a new query object.
        The query string is set in the constructor, and other options have
        setter functions.
        """

        self._query_string = query_string
        self._offset = 0
        self._num = 10
        self._no_content = False
        self._no_stopwords = False
        self._fields = None
        self._verbatim = False
        self._with_payloads = False
        self._with_scores = False
        self._scorer = False
        self._filters = list()
        self._ids = None
        self._slop = -1
        self._timeout = None
        self._in_order = False
        self._sortby = None
        self._return_fields = []
        self._summarize_fields = []
        self._highlight_fields = []
        self._language = None
        self._expander = None
        self._dialect = None

    def query_string(self):
        """Return the query string of this query only."""
        return self._query_string

    def limit_ids(self, *ids):
        """Limit the results to a specific set of pre-known document
        ids of any length."""
        self._ids = ids
        return self

    def return_fields(self, *fields):
        """Add fields to return fields."""
        self._return_fields += fields
        return self

    def return_field(self, field, as_field=None):
        """Add field to return fields (Optional: add 'AS' name
        to the field)."""
        self._return_fields.append(field)
        if as_field is not None:
            self._return_fields += ("AS", as_field)
        return self

    def _mk_field_list(self, fields):
        if not fields:
            return []
        return [fields] if isinstance(fields, str) else list(fields)

    def summarize(self, fields=None, context_len=None, num_frags=None, sep=None):
        """
        Return an abridged format of the field, containing only the segments of
        the field which contain the matching term(s).

        If `fields` is specified, then only the mentioned fields are
        summarized; otherwise all results are summarized.

        Server side defaults are used for each option (except `fields`)
        if not specified

        - **fields** List of fields to summarize. All fields are summarized
        if not specified
        - **context_len** Amount of context to include with each fragment
        - **num_frags** Number of fragments per document
        - **sep** Separator string to separate fragments
        """
        args = ["SUMMARIZE"]
        fields = self._mk_field_list(fields)
        if fields:
            args += ["FIELDS", str(len(fields))] + fields

        if context_len is not None:
            args += ["LEN", str(context_len)]
        if num_frags is not None:
            args += ["FRAGS", str(num_frags)]
        if sep is not None:
            args += ["SEPARATOR", sep]

        self._summarize_fields = args
        return self

    def highlight(self, fields=None, tags=None):
        """
        Apply specified markup to matched term(s) within the returned field(s).

        - **fields** If specified then only those mentioned fields are
        highlighted, otherwise all fields are highlighted
        - **tags** A list of two strings to surround the match.
        """
        args = ["HIGHLIGHT"]
        fields = self._mk_field_list(fields)
        if fields:
            args += ["FIELDS", str(len(fields))] + fields
        if tags:
            args += ["TAGS"] + list(tags)

        self._highlight_fields = args
        return self

    def language(self, language):
        """
        Analyze the query as being in the specified language.

        :param language: The language (e.g. `chinese` or `english`)
        """
        self._language = language
        return self

    def slop(self, slop):
        """Allow a maximum of N intervening non matched terms between
        phrase terms (0 means exact phrase).
        """
        self._slop = slop
        return self

    def timeout(self, timeout):
        """overrides the timeout parameter of the module"""
        self._timeout = timeout
        return self

    def in_order(self):
        """
        Match only documents where the query terms appear in
        the same order in the document.
        i.e. for the query "hello world", we do not match "world hello"
        """
        self._in_order = True
        return self

    def scorer(self, scorer):
        """
        Use a different scoring function to evaluate document relevance.
        Default is `TFIDF`.

        :param scorer: The scoring function to use
                       (e.g. `TFIDF.DOCNORM` or `BM25`)
        """
        self._scorer = scorer
        return self

    def get_args(self):
        """Format the redis arguments for this query and return them."""
        args = [self._query_string]
        args += self._get_args_tags()
        args += self._summarize_fields + self._highlight_fields
        args += ["LIMIT", self._offset, self._num]
        return args

    def _get_args_tags(self):
        args = []
        if self._no_content:
            args.append("NOCONTENT")
        if self._fields:
            args.append("INFIELDS")
            args.append(len(self._fields))
            args += self._fields
        if self._verbatim:
            args.append("VERBATIM")
        if self._no_stopwords:
            args.append("NOSTOPWORDS")
        if self._filters:
            for flt in self._filters:
                if not isinstance(flt, Filter):
                    raise AttributeError("Did not receive a Filter object.")
                args += flt.args
        if self._with_payloads:
            args.append("WITHPAYLOADS")
        if self._scorer:
            args += ["SCORER", self._scorer]
        if self._with_scores:
            args.append("WITHSCORES")
        if self._ids:
            args.append("INKEYS")
            args.append(len(self._ids))
            args += self._ids
        if self._slop >= 0:
            args += ["SLOP", self._slop]
        if self._timeout:
            args += ["TIMEOUT", self._timeout]
        if self._in_order:
            args.append("INORDER")
        if self._return_fields:
            args.append("RETURN")
            args.append(len(self._return_fields))
            args += self._return_fields
        if self._sortby:
            if not isinstance(self._sortby, SortbyField):
                raise AttributeError("Did not receive a SortByField.")
            args.append("SORTBY")
            args += self._sortby.args
        if self._language:
            args += ["LANGUAGE", self._language]
        if self._expander:
            args += ["EXPANDER", self._expander]
        if self._dialect:
            args += ["DIALECT", self._dialect]

        return args

    def paging(self, offset, num):
        """
        Set the paging for the query (defaults to 0..10).

        - **offset**: Paging offset for the results. Defaults to 0
        - **num**: How many results do we want
        """
        self._offset = offset
        self._num = num
        return self

    def verbatim(self):
        """Set the query to be verbatim, i.e. use no query expansion
        or stemming.
        """
        self._verbatim = True
        return self

    def no_content(self):
        """Set the query to only return ids and not the document content."""
        self._no_content = True
        return self

    def no_stopwords(self):
        """
        Prevent the query from being filtered for stopwords.
        Only useful in very big queries that you are certain contain
        no stopwords.
        """
        self._no_stopwords = True
        return self

    def with_payloads(self):
        """Ask the engine to return document payloads."""
        self._with_payloads = True
        return self

    def with_scores(self):
        """Ask the engine to return document search scores."""
        self._with_scores = True
        return self

    def limit_fields(self, *fields):
        """
        Limit the search to specific TEXT fields only.

        - **fields**: A list of strings, case sensitive field names
        from the defined schema.
        """
        self._fields = fields
        return self

    def add_filter(self, flt):
        """
        Add a numeric or geo filter to the query.
        **Currently only one of each filter is supported by the engine**

        - **flt**: A NumericFilter or GeoFilter object, used on a
        corresponding field
        """

        self._filters.append(flt)
        return self

    def sort_by(self, field, asc=True):
        """
        Add a sortby field to the query.

        - **field** - the name of the field to sort by
        - **asc** - when `True`, sorting will be done in asceding order
        """
        self._sortby = SortbyField(field, asc)
        return self

    def expander(self, expander):
        """
        Add a expander field to the query.

        - **expander** - the name of the expander
        """
        self._expander = expander
        return self

    def dialect(self, dialect: int) -> "Query":
        """
        Add a dialect field to the query.

        - **dialect** - dialect version to execute the query under
        """
        self._dialect = dialect
        return self


class Filter:
    def __init__(self, keyword, field, *args):
        self.args = [keyword, field] + list(args)


class NumericFilter(Filter):
    INF = "+inf"
    NEG_INF = "-inf"

    def __init__(self, field, minval, maxval, minExclusive=False, maxExclusive=False):
        args = [
            minval if not minExclusive else f"({minval}",
            maxval if not maxExclusive else f"({maxval}",
        ]

        Filter.__init__(self, "FILTER", field, *args)


class GeoFilter(Filter):
    METERS = "m"
    KILOMETERS = "km"
    FEET = "ft"
    MILES = "mi"

    def __init__(self, field, lon, lat, radius, unit=KILOMETERS):
        Filter.__init__(self, "GEOFILTER", field, lon, lat, radius, unit)


class SortbyField:
    def __init__(self, field, asc=True):
        self.args = [field, "ASC" if asc else "DESC"]
