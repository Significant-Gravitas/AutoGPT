from typing import List

from redis import DataError


class Field:

    NUMERIC = "NUMERIC"
    TEXT = "TEXT"
    WEIGHT = "WEIGHT"
    GEO = "GEO"
    TAG = "TAG"
    VECTOR = "VECTOR"
    SORTABLE = "SORTABLE"
    NOINDEX = "NOINDEX"
    AS = "AS"

    def __init__(
        self,
        name: str,
        args: List[str] = None,
        sortable: bool = False,
        no_index: bool = False,
        as_name: str = None,
    ):
        if args is None:
            args = []
        self.name = name
        self.args = args
        self.args_suffix = list()
        self.as_name = as_name

        if sortable:
            self.args_suffix.append(Field.SORTABLE)
        if no_index:
            self.args_suffix.append(Field.NOINDEX)

        if no_index and not sortable:
            raise ValueError("Non-Sortable non-Indexable fields are ignored")

    def append_arg(self, value):
        self.args.append(value)

    def redis_args(self):
        args = [self.name]
        if self.as_name:
            args += [self.AS, self.as_name]
        args += self.args
        args += self.args_suffix
        return args


class TextField(Field):
    """
    TextField is used to define a text field in a schema definition
    """

    NOSTEM = "NOSTEM"
    PHONETIC = "PHONETIC"

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        no_stem: bool = False,
        phonetic_matcher: str = None,
        withsuffixtrie: bool = False,
        **kwargs,
    ):
        Field.__init__(self, name, args=[Field.TEXT, Field.WEIGHT, weight], **kwargs)

        if no_stem:
            Field.append_arg(self, self.NOSTEM)
        if phonetic_matcher and phonetic_matcher in [
            "dm:en",
            "dm:fr",
            "dm:pt",
            "dm:es",
        ]:
            Field.append_arg(self, self.PHONETIC)
            Field.append_arg(self, phonetic_matcher)
        if withsuffixtrie:
            Field.append_arg(self, "WITHSUFFIXTRIE")


class NumericField(Field):
    """
    NumericField is used to define a numeric field in a schema definition
    """

    def __init__(self, name: str, **kwargs):
        Field.__init__(self, name, args=[Field.NUMERIC], **kwargs)


class GeoField(Field):
    """
    GeoField is used to define a geo-indexing field in a schema definition
    """

    def __init__(self, name: str, **kwargs):
        Field.__init__(self, name, args=[Field.GEO], **kwargs)


class TagField(Field):
    """
    TagField is a tag-indexing field with simpler compression and tokenization.
    See http://redisearch.io/Tags/
    """

    SEPARATOR = "SEPARATOR"
    CASESENSITIVE = "CASESENSITIVE"

    def __init__(
        self,
        name: str,
        separator: str = ",",
        case_sensitive: bool = False,
        withsuffixtrie: bool = False,
        **kwargs,
    ):
        args = [Field.TAG, self.SEPARATOR, separator]
        if case_sensitive:
            args.append(self.CASESENSITIVE)
        if withsuffixtrie:
            args.append("WITHSUFFIXTRIE")

        Field.__init__(self, name, args=args, **kwargs)


class VectorField(Field):
    """
    Allows vector similarity queries against the value in this attribute.
    See https://oss.redis.com/redisearch/Vectors/#vector_fields.
    """

    def __init__(self, name: str, algorithm: str, attributes: dict, **kwargs):
        """
        Create Vector Field. Notice that Vector cannot have sortable or no_index tag,
        although it's also a Field.

        ``name`` is the name of the field.

        ``algorithm`` can be "FLAT" or "HNSW".

        ``attributes`` each algorithm can have specific attributes. Some of them
        are mandatory and some of them are optional. See
        https://oss.redis.com/redisearch/master/Vectors/#specific_creation_attributes_per_algorithm
        for more information.
        """
        sort = kwargs.get("sortable", False)
        noindex = kwargs.get("no_index", False)

        if sort or noindex:
            raise DataError("Cannot set 'sortable' or 'no_index' in Vector fields.")

        if algorithm.upper() not in ["FLAT", "HNSW"]:
            raise DataError(
                "Realtime vector indexing supporting 2 Indexing Methods:"
                "'FLAT' and 'HNSW'."
            )

        attr_li = []

        for key, value in attributes.items():
            attr_li.extend([key, value])

        Field.__init__(
            self, name, args=[Field.VECTOR, algorithm, len(attr_li), *attr_li], **kwargs
        )
