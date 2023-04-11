def tags(*t):
    """
    Indicate that the values should be matched to a tag field

    ### Parameters

    - **t**: Tags to search for
    """
    if not t:
        raise ValueError("At least one tag must be specified")
    return TagValue(*t)


def between(a, b, inclusive_min=True, inclusive_max=True):
    """
    Indicate that value is a numeric range
    """
    return RangeValue(a, b, inclusive_min=inclusive_min, inclusive_max=inclusive_max)


def equal(n):
    """
    Match a numeric value
    """
    return between(n, n)


def lt(n):
    """
    Match any value less than n
    """
    return between(None, n, inclusive_max=False)


def le(n):
    """
    Match any value less or equal to n
    """
    return between(None, n, inclusive_max=True)


def gt(n):
    """
    Match any value greater than n
    """
    return between(n, None, inclusive_min=False)


def ge(n):
    """
    Match any value greater or equal to n
    """
    return between(n, None, inclusive_min=True)


def geo(lat, lon, radius, unit="km"):
    """
    Indicate that value is a geo region
    """
    return GeoValue(lat, lon, radius, unit)


class Value:
    @property
    def combinable(self):
        """
        Whether this type of value may be combined with other values
        for the same field. This makes the filter potentially more efficient
        """
        return False

    @staticmethod
    def make_value(v):
        """
        Convert an object to a value, if it is not a value already
        """
        if isinstance(v, Value):
            return v
        return ScalarValue(v)

    def to_string(self):
        raise NotImplementedError()

    def __str__(self):
        return self.to_string()


class RangeValue(Value):
    combinable = False

    def __init__(self, a, b, inclusive_min=False, inclusive_max=False):
        if a is None:
            a = "-inf"
        if b is None:
            b = "inf"
        self.range = [str(a), str(b)]
        self.inclusive_min = inclusive_min
        self.inclusive_max = inclusive_max

    def to_string(self):
        return "[{1}{0[0]} {2}{0[1]}]".format(
            self.range,
            "(" if not self.inclusive_min else "",
            "(" if not self.inclusive_max else "",
        )


class ScalarValue(Value):
    combinable = True

    def __init__(self, v):
        self.v = str(v)

    def to_string(self):
        return self.v


class TagValue(Value):
    combinable = False

    def __init__(self, *tags):
        self.tags = tags

    def to_string(self):
        return "{" + " | ".join(str(t) for t in self.tags) + "}"


class GeoValue(Value):
    def __init__(self, lon, lat, radius, unit="km"):
        self.lon = lon
        self.lat = lat
        self.radius = radius
        self.unit = unit

    def to_string(self):
        return f"[{self.lon} {self.lat} {self.radius} {self.unit}]"


class Node:
    def __init__(self, *children, **kwparams):
        """
        Create a node

        ### Parameters

        - **children**: One or more sub-conditions. These can be additional
            `intersect`, `disjunct`, `union`, `optional`, or any other `Node`
            type.

            The semantics of multiple conditions are dependent on the type of
            query. For an `intersection` node, this amounts to a logical AND,
            for a `union` node, this amounts to a logical `OR`.

        - **kwparams**: key-value parameters. Each key is the name of a field,
            and the value should be a field value. This can be one of the
            following:

            - Simple string (for text field matches)
            - value returned by one of the helper functions
            - list of either a string or a value


        ### Examples

        Field `num` should be between 1 and 10
        ```
        intersect(num=between(1, 10)
        ```

        Name can either be `bob` or `john`

        ```
        union(name=("bob", "john"))
        ```

        Don't select countries in Israel, Japan, or US

        ```
        disjunct_union(country=("il", "jp", "us"))
        ```
        """

        self.params = []

        kvparams = {}
        for k, v in kwparams.items():
            curvals = kvparams.setdefault(k, [])
            if isinstance(v, (str, int, float)):
                curvals.append(Value.make_value(v))
            elif isinstance(v, Value):
                curvals.append(v)
            else:
                curvals.extend(Value.make_value(subv) for subv in v)

        self.params += [Node.to_node(p) for p in children]

        for k, v in kvparams.items():
            self.params.extend(self.join_fields(k, v))

    def join_fields(self, key, vals):
        if len(vals) == 1:
            return [BaseNode(f"@{key}:{vals[0].to_string()}")]
        if not vals[0].combinable:
            return [BaseNode(f"@{key}:{v.to_string()}") for v in vals]
        s = BaseNode(f"@{key}:({self.JOINSTR.join(v.to_string() for v in vals)})")
        return [s]

    @classmethod
    def to_node(cls, obj):  # noqa
        if isinstance(obj, Node):
            return obj
        return BaseNode(obj)

    @property
    def JOINSTR(self):
        raise NotImplementedError()

    def to_string(self, with_parens=None):
        with_parens = self._should_use_paren(with_parens)
        pre, post = ("(", ")") if with_parens else ("", "")
        return f"{pre}{self.JOINSTR.join(n.to_string() for n in self.params)}{post}"

    def _should_use_paren(self, optval):
        if optval is not None:
            return optval
        return len(self.params) > 1

    def __str__(self):
        return self.to_string()


class BaseNode(Node):
    def __init__(self, s):
        super().__init__()
        self.s = str(s)

    def to_string(self, with_parens=None):
        return self.s


class IntersectNode(Node):
    """
    Create an intersection node. All children need to be satisfied in order for
    this node to evaluate as true
    """

    JOINSTR = " "


class UnionNode(Node):
    """
    Create a union node. Any of the children need to be satisfied in order for
    this node to evaluate as true
    """

    JOINSTR = "|"


class DisjunctNode(IntersectNode):
    """
    Create a disjunct node. In order for this node to be true, all of its
    children must evaluate to false
    """

    def to_string(self, with_parens=None):
        with_parens = self._should_use_paren(with_parens)
        ret = super().to_string(with_parens=False)
        if with_parens:
            return "(-" + ret + ")"
        else:
            return "-" + ret


class DistjunctUnion(DisjunctNode):
    """
    This node is true if *all* of its children are false. This is equivalent to
    ```
    disjunct(union(...))
    ```
    """

    JOINSTR = "|"


class OptionalNode(IntersectNode):
    """
    Create an optional node. If this nodes evaluates to true, then the document
    will be rated higher in score/rank.
    """

    def to_string(self, with_parens=None):
        with_parens = self._should_use_paren(with_parens)
        ret = super().to_string(with_parens=False)
        if with_parens:
            return "(~" + ret + ")"
        else:
            return "~" + ret


def intersect(*args, **kwargs):
    return IntersectNode(*args, **kwargs)


def union(*args, **kwargs):
    return UnionNode(*args, **kwargs)


def disjunct(*args, **kwargs):
    return DisjunctNode(*args, **kwargs)


def disjunct_union(*args, **kwargs):
    return DistjunctUnion(*args, **kwargs)


def querystring(*args, **kwargs):
    return intersect(*args, **kwargs).to_string()
