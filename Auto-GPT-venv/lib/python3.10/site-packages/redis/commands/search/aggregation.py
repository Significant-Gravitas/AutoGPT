FIELDNAME = object()


class Limit:
    def __init__(self, offset=0, count=0):
        self.offset = offset
        self.count = count

    def build_args(self):
        if self.count:
            return ["LIMIT", str(self.offset), str(self.count)]
        else:
            return []


class Reducer:
    """
    Base reducer object for all reducers.

    See the `redisearch.reducers` module for the actual reducers.
    """

    NAME = None

    def __init__(self, *args):
        self._args = args
        self._field = None
        self._alias = None

    def alias(self, alias):
        """
        Set the alias for this reducer.

        ### Parameters

        - **alias**: The value of the alias for this reducer. If this is the
            special value `aggregation.FIELDNAME` then this reducer will be
            aliased using the same name as the field upon which it operates.
            Note that using `FIELDNAME` is only possible on reducers which
            operate on a single field value.

        This method returns the `Reducer` object making it suitable for
        chaining.
        """
        if alias is FIELDNAME:
            if not self._field:
                raise ValueError("Cannot use FIELDNAME alias with no field")
            # Chop off initial '@'
            alias = self._field[1:]
        self._alias = alias
        return self

    @property
    def args(self):
        return self._args


class SortDirection:
    """
    This special class is used to indicate sort direction.
    """

    DIRSTRING = None

    def __init__(self, field):
        self.field = field


class Asc(SortDirection):
    """
    Indicate that the given field should be sorted in ascending order
    """

    DIRSTRING = "ASC"


class Desc(SortDirection):
    """
    Indicate that the given field should be sorted in descending order
    """

    DIRSTRING = "DESC"


class AggregateRequest:
    """
    Aggregation request which can be passed to `Client.aggregate`.
    """

    def __init__(self, query="*"):
        """
        Create an aggregation request. This request may then be passed to
        `client.aggregate()`.

        In order for the request to be usable, it must contain at least one
        group.

        - **query** Query string for filtering records.

        All member methods (except `build_args()`)
        return the object itself, making them useful for chaining.
        """
        self._query = query
        self._aggregateplan = []
        self._loadfields = []
        self._loadall = False
        self._max = 0
        self._with_schema = False
        self._verbatim = False
        self._cursor = []
        self._dialect = None

    def load(self, *fields):
        """
        Indicate the fields to be returned in the response. These fields are
        returned in addition to any others implicitly specified.

        ### Parameters

        - **fields**: If fields not specified, all the fields will be loaded.
        Otherwise, fields should be given in the format of `@field`.
        """
        if fields:
            self._loadfields.extend(fields)
        else:
            self._loadall = True
        return self

    def group_by(self, fields, *reducers):
        """
        Specify by which fields to group the aggregation.

        ### Parameters

        - **fields**: Fields to group by. This can either be a single string,
            or a list of strings. both cases, the field should be specified as
            `@field`.
        - **reducers**: One or more reducers. Reducers may be found in the
            `aggregation` module.
        """
        fields = [fields] if isinstance(fields, str) else fields
        reducers = [reducers] if isinstance(reducers, Reducer) else reducers

        ret = ["GROUPBY", str(len(fields)), *fields]
        for reducer in reducers:
            ret += ["REDUCE", reducer.NAME, str(len(reducer.args))]
            ret.extend(reducer.args)
            if reducer._alias is not None:
                ret += ["AS", reducer._alias]

        self._aggregateplan.extend(ret)
        return self

    def apply(self, **kwexpr):
        """
        Specify one or more projection expressions to add to each result

        ### Parameters

        - **kwexpr**: One or more key-value pairs for a projection. The key is
            the alias for the projection, and the value is the projection
            expression itself, for example `apply(square_root="sqrt(@foo)")`
        """
        for alias, expr in kwexpr.items():
            ret = ["APPLY", expr]
            if alias is not None:
                ret += ["AS", alias]
            self._aggregateplan.extend(ret)

        return self

    def limit(self, offset, num):
        """
        Sets the limit for the most recent group or query.

        If no group has been defined yet (via `group_by()`) then this sets
        the limit for the initial pool of results from the query. Otherwise,
        this limits the number of items operated on from the previous group.

        Setting a limit on the initial search results may be useful when
        attempting to execute an aggregation on a sample of a large data set.

        ### Parameters

        - **offset**: Result offset from which to begin paging
        - **num**: Number of results to return


        Example of sorting the initial results:

        ```
        AggregateRequest("@sale_amount:[10000, inf]")\
            .limit(0, 10)\
            .group_by("@state", r.count())
        ```

        Will only group by the states found in the first 10 results of the
        query `@sale_amount:[10000, inf]`. On the other hand,

        ```
        AggregateRequest("@sale_amount:[10000, inf]")\
            .limit(0, 1000)\
            .group_by("@state", r.count()\
            .limit(0, 10)
        ```

        Will group all the results matching the query, but only return the
        first 10 groups.

        If you only wish to return a *top-N* style query, consider using
        `sort_by()` instead.

        """
        _limit = Limit(offset, num)
        self._aggregateplan.extend(_limit.build_args())
        return self

    def sort_by(self, *fields, **kwargs):
        """
        Indicate how the results should be sorted. This can also be used for
        *top-N* style queries

        ### Parameters

        - **fields**: The fields by which to sort. This can be either a single
            field or a list of fields. If you wish to specify order, you can
            use the `Asc` or `Desc` wrapper classes.
        - **max**: Maximum number of results to return. This can be
            used instead of `LIMIT` and is also faster.


        Example of sorting by `foo` ascending and `bar` descending:

        ```
        sort_by(Asc("@foo"), Desc("@bar"))
        ```

        Return the top 10 customers:

        ```
        AggregateRequest()\
            .group_by("@customer", r.sum("@paid").alias(FIELDNAME))\
            .sort_by(Desc("@paid"), max=10)
        ```
        """
        if isinstance(fields, (str, SortDirection)):
            fields = [fields]

        fields_args = []
        for f in fields:
            if isinstance(f, SortDirection):
                fields_args += [f.field, f.DIRSTRING]
            else:
                fields_args += [f]

        ret = ["SORTBY", str(len(fields_args))]
        ret.extend(fields_args)
        max = kwargs.get("max", 0)
        if max > 0:
            ret += ["MAX", str(max)]

        self._aggregateplan.extend(ret)
        return self

    def filter(self, expressions):
        """
        Specify filter for post-query results using predicates relating to
        values in the result set.

        ### Parameters

        - **fields**: Fields to group by. This can either be a single string,
            or a list of strings.
        """
        if isinstance(expressions, str):
            expressions = [expressions]

        for expression in expressions:
            self._aggregateplan.extend(["FILTER", expression])

        return self

    def with_schema(self):
        """
        If set, the `schema` property will contain a list of `[field, type]`
        entries in the result object.
        """
        self._with_schema = True
        return self

    def verbatim(self):
        self._verbatim = True
        return self

    def cursor(self, count=0, max_idle=0.0):
        args = ["WITHCURSOR"]
        if count:
            args += ["COUNT", str(count)]
        if max_idle:
            args += ["MAXIDLE", str(max_idle * 1000)]
        self._cursor = args
        return self

    def build_args(self):
        # @foo:bar ...
        ret = [self._query]

        if self._with_schema:
            ret.append("WITHSCHEMA")

        if self._verbatim:
            ret.append("VERBATIM")

        if self._cursor:
            ret += self._cursor

        if self._loadall:
            ret.append("LOAD")
            ret.append("*")
        elif self._loadfields:
            ret.append("LOAD")
            ret.append(str(len(self._loadfields)))
            ret.extend(self._loadfields)

        if self._dialect:
            ret.extend(["DIALECT", self._dialect])

        ret.extend(self._aggregateplan)

        return ret

    def dialect(self, dialect):
        """
        Add a dialect field to the aggregate command.

        - **dialect** - dialect version to execute the query under
        """
        self._dialect = dialect
        return self


class Cursor:
    def __init__(self, cid):
        self.cid = cid
        self.max_idle = 0
        self.count = 0

    def build_args(self):
        args = [str(self.cid)]
        if self.max_idle:
            args += ["MAXIDLE", str(self.max_idle)]
        if self.count:
            args += ["COUNT", str(self.count)]
        return args


class AggregateResult:
    def __init__(self, rows, cursor, schema):
        self.rows = rows
        self.cursor = cursor
        self.schema = schema

    def __repr__(self):
        cid = self.cursor.cid if self.cursor else -1
        return (
            f"<{self.__class__.__name__} at 0x{id(self):x} "
            f"Rows={len(self.rows)}, Cursor={cid}>"
        )
