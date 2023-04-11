import itertools
import time
from typing import Dict, Optional, Union

from redis.client import Pipeline
from redis.utils import deprecated_function

from ..helpers import parse_to_dict
from ._util import to_string
from .aggregation import AggregateRequest, AggregateResult, Cursor
from .document import Document
from .query import Query
from .result import Result
from .suggestion import SuggestionParser

NUMERIC = "NUMERIC"

CREATE_CMD = "FT.CREATE"
ALTER_CMD = "FT.ALTER"
SEARCH_CMD = "FT.SEARCH"
ADD_CMD = "FT.ADD"
ADDHASH_CMD = "FT.ADDHASH"
DROP_CMD = "FT.DROP"
DROPINDEX_CMD = "FT.DROPINDEX"
EXPLAIN_CMD = "FT.EXPLAIN"
EXPLAINCLI_CMD = "FT.EXPLAINCLI"
DEL_CMD = "FT.DEL"
AGGREGATE_CMD = "FT.AGGREGATE"
PROFILE_CMD = "FT.PROFILE"
CURSOR_CMD = "FT.CURSOR"
SPELLCHECK_CMD = "FT.SPELLCHECK"
DICT_ADD_CMD = "FT.DICTADD"
DICT_DEL_CMD = "FT.DICTDEL"
DICT_DUMP_CMD = "FT.DICTDUMP"
GET_CMD = "FT.GET"
MGET_CMD = "FT.MGET"
CONFIG_CMD = "FT.CONFIG"
TAGVALS_CMD = "FT.TAGVALS"
ALIAS_ADD_CMD = "FT.ALIASADD"
ALIAS_UPDATE_CMD = "FT.ALIASUPDATE"
ALIAS_DEL_CMD = "FT.ALIASDEL"
INFO_CMD = "FT.INFO"
SUGADD_COMMAND = "FT.SUGADD"
SUGDEL_COMMAND = "FT.SUGDEL"
SUGLEN_COMMAND = "FT.SUGLEN"
SUGGET_COMMAND = "FT.SUGGET"
SYNUPDATE_CMD = "FT.SYNUPDATE"
SYNDUMP_CMD = "FT.SYNDUMP"

NOOFFSETS = "NOOFFSETS"
NOFIELDS = "NOFIELDS"
NOHL = "NOHL"
NOFREQS = "NOFREQS"
MAXTEXTFIELDS = "MAXTEXTFIELDS"
TEMPORARY = "TEMPORARY"
STOPWORDS = "STOPWORDS"
SKIPINITIALSCAN = "SKIPINITIALSCAN"
WITHSCORES = "WITHSCORES"
FUZZY = "FUZZY"
WITHPAYLOADS = "WITHPAYLOADS"


class SearchCommands:
    """Search commands."""

    def batch_indexer(self, chunk_size=100):
        """
        Create a new batch indexer from the client with a given chunk size
        """
        return self.BatchIndexer(self, chunk_size=chunk_size)

    def create_index(
        self,
        fields,
        no_term_offsets=False,
        no_field_flags=False,
        stopwords=None,
        definition=None,
        max_text_fields=False,
        temporary=None,
        no_highlight=False,
        no_term_frequencies=False,
        skip_initial_scan=False,
    ):
        """
        Create the search index. The index must not already exist.

        ### Parameters:

        - **fields**: a list of TextField or NumericField objects
        - **no_term_offsets**: If true, we will not save term offsets in
        the index
        - **no_field_flags**: If true, we will not save field flags that
        allow searching in specific fields
        - **stopwords**: If not None, we create the index with this custom
        stopword list. The list can be empty
        - **max_text_fields**: If true, we will encode indexes as if there
        were more than 32 text fields which allows you to add additional
        fields (beyond 32).
        - **temporary**: Create a lightweight temporary index which will
        expire after the specified period of inactivity (in seconds). The
        internal idle timer is reset whenever the index is searched or added to.
        - **no_highlight**: If true, disabling highlighting support.
        Also implied by no_term_offsets.
        - **no_term_frequencies**: If true, we avoid saving the term frequencies
        in the index.
        - **skip_initial_scan**: If true, we do not scan and index.

        For more information see `FT.CREATE <https://redis.io/commands/ft.create>`_.
        """  # noqa

        args = [CREATE_CMD, self.index_name]
        if definition is not None:
            args += definition.args
        if max_text_fields:
            args.append(MAXTEXTFIELDS)
        if temporary is not None and isinstance(temporary, int):
            args.append(TEMPORARY)
            args.append(temporary)
        if no_term_offsets:
            args.append(NOOFFSETS)
        if no_highlight:
            args.append(NOHL)
        if no_field_flags:
            args.append(NOFIELDS)
        if no_term_frequencies:
            args.append(NOFREQS)
        if skip_initial_scan:
            args.append(SKIPINITIALSCAN)
        if stopwords is not None and isinstance(stopwords, (list, tuple, set)):
            args += [STOPWORDS, len(stopwords)]
            if len(stopwords) > 0:
                args += list(stopwords)

        args.append("SCHEMA")
        try:
            args += list(itertools.chain(*(f.redis_args() for f in fields)))
        except TypeError:
            args += fields.redis_args()

        return self.execute_command(*args)

    def alter_schema_add(self, fields):
        """
        Alter the existing search index by adding new fields. The index
        must already exist.

        ### Parameters:

        - **fields**: a list of Field objects to add for the index

        For more information see `FT.ALTER <https://redis.io/commands/ft.alter>`_.
        """  # noqa

        args = [ALTER_CMD, self.index_name, "SCHEMA", "ADD"]
        try:
            args += list(itertools.chain(*(f.redis_args() for f in fields)))
        except TypeError:
            args += fields.redis_args()

        return self.execute_command(*args)

    def dropindex(self, delete_documents=False):
        """
        Drop the index if it exists.
        Replaced `drop_index` in RediSearch 2.0.
        Default behavior was changed to not delete the indexed documents.

        ### Parameters:

        - **delete_documents**: If `True`, all documents will be deleted.

        For more information see `FT.DROPINDEX <https://redis.io/commands/ft.dropindex>`_.
        """  # noqa
        delete_str = "DD" if delete_documents else ""
        return self.execute_command(DROPINDEX_CMD, self.index_name, delete_str)

    def _add_document(
        self,
        doc_id,
        conn=None,
        nosave=False,
        score=1.0,
        payload=None,
        replace=False,
        partial=False,
        language=None,
        no_create=False,
        **fields,
    ):
        """
        Internal add_document used for both batch and single doc indexing
        """

        if partial or no_create:
            replace = True

        args = [ADD_CMD, self.index_name, doc_id, score]
        if nosave:
            args.append("NOSAVE")
        if payload is not None:
            args.append("PAYLOAD")
            args.append(payload)
        if replace:
            args.append("REPLACE")
            if partial:
                args.append("PARTIAL")
            if no_create:
                args.append("NOCREATE")
        if language:
            args += ["LANGUAGE", language]
        args.append("FIELDS")
        args += list(itertools.chain(*fields.items()))

        if conn is not None:
            return conn.execute_command(*args)

        return self.execute_command(*args)

    def _add_document_hash(
        self, doc_id, conn=None, score=1.0, language=None, replace=False
    ):
        """
        Internal add_document_hash used for both batch and single doc indexing
        """

        args = [ADDHASH_CMD, self.index_name, doc_id, score]

        if replace:
            args.append("REPLACE")

        if language:
            args += ["LANGUAGE", language]

        if conn is not None:
            return conn.execute_command(*args)

        return self.execute_command(*args)

    @deprecated_function(
        version="2.0.0", reason="deprecated since redisearch 2.0, call hset instead"
    )
    def add_document(
        self,
        doc_id,
        nosave=False,
        score=1.0,
        payload=None,
        replace=False,
        partial=False,
        language=None,
        no_create=False,
        **fields,
    ):
        """
        Add a single document to the index.

        ### Parameters

        - **doc_id**: the id of the saved document.
        - **nosave**: if set to true, we just index the document, and don't
                      save a copy of it. This means that searches will just
                      return ids.
        - **score**: the document ranking, between 0.0 and 1.0
        - **payload**: optional inner-index payload we can save for fast
        i              access in scoring functions
        - **replace**: if True, and the document already is in the index,
        we perform an update and reindex the document
        - **partial**: if True, the fields specified will be added to the
                       existing document.
                       This has the added benefit that any fields specified
                       with `no_index`
                       will not be reindexed again. Implies `replace`
        - **language**: Specify the language used for document tokenization.
        - **no_create**: if True, the document is only updated and reindexed
                         if it already exists.
                         If the document does not exist, an error will be
                         returned. Implies `replace`
        - **fields** kwargs dictionary of the document fields to be saved
                         and/or indexed.
                     NOTE: Geo points shoule be encoded as strings of "lon,lat"
        """  # noqa
        return self._add_document(
            doc_id,
            conn=None,
            nosave=nosave,
            score=score,
            payload=payload,
            replace=replace,
            partial=partial,
            language=language,
            no_create=no_create,
            **fields,
        )

    @deprecated_function(
        version="2.0.0", reason="deprecated since redisearch 2.0, call hset instead"
    )
    def add_document_hash(self, doc_id, score=1.0, language=None, replace=False):
        """
        Add a hash document to the index.

        ### Parameters

        - **doc_id**: the document's id. This has to be an existing HASH key
                      in Redis that will hold the fields the index needs.
        - **score**:  the document ranking, between 0.0 and 1.0
        - **replace**: if True, and the document already is in the index, we
                      perform an update and reindex the document
        - **language**: Specify the language used for document tokenization.
        """  # noqa
        return self._add_document_hash(
            doc_id, conn=None, score=score, language=language, replace=replace
        )

    def delete_document(self, doc_id, conn=None, delete_actual_document=False):
        """
        Delete a document from index
        Returns 1 if the document was deleted, 0 if not

        ### Parameters

        - **delete_actual_document**: if set to True, RediSearch also delete
                                      the actual document if it is in the index
        """  # noqa
        args = [DEL_CMD, self.index_name, doc_id]
        if delete_actual_document:
            args.append("DD")

        if conn is not None:
            return conn.execute_command(*args)

        return self.execute_command(*args)

    def load_document(self, id):
        """
        Load a single document by id
        """
        fields = self.client.hgetall(id)
        f2 = {to_string(k): to_string(v) for k, v in fields.items()}
        fields = f2

        try:
            del fields["id"]
        except KeyError:
            pass

        return Document(id=id, **fields)

    def get(self, *ids):
        """
        Returns the full contents of multiple documents.

        ### Parameters

        - **ids**: the ids of the saved documents.

        """

        return self.execute_command(MGET_CMD, self.index_name, *ids)

    def info(self):
        """
        Get info an stats about the the current index, including the number of
        documents, memory consumption, etc

        For more information see `FT.INFO <https://redis.io/commands/ft.info>`_.
        """

        res = self.execute_command(INFO_CMD, self.index_name)
        it = map(to_string, res)
        return dict(zip(it, it))

    def get_params_args(
        self, query_params: Union[Dict[str, Union[str, int, float]], None]
    ):
        if query_params is None:
            return []
        args = []
        if len(query_params) > 0:
            args.append("params")
            args.append(len(query_params) * 2)
            for key, value in query_params.items():
                args.append(key)
                args.append(value)
        return args

    def _mk_query_args(self, query, query_params: Dict[str, Union[str, int, float]]):
        args = [self.index_name]

        if isinstance(query, str):
            # convert the query from a text to a query object
            query = Query(query)
        if not isinstance(query, Query):
            raise ValueError(f"Bad query type {type(query)}")

        args += query.get_args()
        args += self.get_params_args(query_params)

        return args, query

    def search(
        self,
        query: Union[str, Query],
        query_params: Dict[str, Union[str, int, float]] = None,
    ):
        """
        Search the index for a given query, and return a result of documents

        ### Parameters

        - **query**: the search query. Either a text for simple queries with
                     default parameters, or a Query object for complex queries.
                     See RediSearch's documentation on query format

        For more information see `FT.SEARCH <https://redis.io/commands/ft.search>`_.
        """  # noqa
        args, query = self._mk_query_args(query, query_params=query_params)
        st = time.time()
        res = self.execute_command(SEARCH_CMD, *args)

        if isinstance(res, Pipeline):
            return res

        return Result(
            res,
            not query._no_content,
            duration=(time.time() - st) * 1000.0,
            has_payload=query._with_payloads,
            with_scores=query._with_scores,
        )

    def explain(
        self,
        query: Union[str, Query],
        query_params: Dict[str, Union[str, int, float]] = None,
    ):
        """Returns the execution plan for a complex query.

        For more information see `FT.EXPLAIN <https://redis.io/commands/ft.explain>`_.
        """  # noqa
        args, query_text = self._mk_query_args(query, query_params=query_params)
        return self.execute_command(EXPLAIN_CMD, *args)

    def explain_cli(self, query: Union[str, Query]):  # noqa
        raise NotImplementedError("EXPLAINCLI will not be implemented.")

    def aggregate(
        self,
        query: Union[str, Query],
        query_params: Dict[str, Union[str, int, float]] = None,
    ):
        """
        Issue an aggregation query.

        ### Parameters

        **query**: This can be either an `AggregateRequest`, or a `Cursor`

        An `AggregateResult` object is returned. You can access the rows from
        its `rows` property, which will always yield the rows of the result.

        For more information see `FT.AGGREGATE <https://redis.io/commands/ft.aggregate>`_.
        """  # noqa
        if isinstance(query, AggregateRequest):
            has_cursor = bool(query._cursor)
            cmd = [AGGREGATE_CMD, self.index_name] + query.build_args()
        elif isinstance(query, Cursor):
            has_cursor = True
            cmd = [CURSOR_CMD, "READ", self.index_name] + query.build_args()
        else:
            raise ValueError("Bad query", query)
        cmd += self.get_params_args(query_params)

        raw = self.execute_command(*cmd)
        return self._get_aggregate_result(raw, query, has_cursor)

    def _get_aggregate_result(self, raw, query, has_cursor):
        if has_cursor:
            if isinstance(query, Cursor):
                query.cid = raw[1]
                cursor = query
            else:
                cursor = Cursor(raw[1])
            raw = raw[0]
        else:
            cursor = None

        if isinstance(query, AggregateRequest) and query._with_schema:
            schema = raw[0]
            rows = raw[2:]
        else:
            schema = None
            rows = raw[1:]

        return AggregateResult(rows, cursor, schema)

    def profile(
        self,
        query: Union[str, Query, AggregateRequest],
        limited: bool = False,
        query_params: Optional[Dict[str, Union[str, int, float]]] = None,
    ):
        """
        Performs a search or aggregate command and collects performance
        information.

        ### Parameters

        **query**: This can be either an `AggregateRequest`, `Query` or string.
        **limited**: If set to True, removes details of reader iterator.
        **query_params**: Define one or more value parameters.
        Each parameter has a name and a value.

        """
        st = time.time()
        cmd = [PROFILE_CMD, self.index_name, ""]
        if limited:
            cmd.append("LIMITED")
        cmd.append("QUERY")

        if isinstance(query, AggregateRequest):
            cmd[2] = "AGGREGATE"
            cmd += query.build_args()
        elif isinstance(query, Query):
            cmd[2] = "SEARCH"
            cmd += query.get_args()
            cmd += self.get_params_args(query_params)
        else:
            raise ValueError("Must provide AggregateRequest object or Query object.")

        res = self.execute_command(*cmd)

        if isinstance(query, AggregateRequest):
            result = self._get_aggregate_result(res[0], query, query._cursor)
        else:
            result = Result(
                res[0],
                not query._no_content,
                duration=(time.time() - st) * 1000.0,
                has_payload=query._with_payloads,
                with_scores=query._with_scores,
            )

        return result, parse_to_dict(res[1])

    def spellcheck(self, query, distance=None, include=None, exclude=None):
        """
        Issue a spellcheck query

        ### Parameters

        **query**: search query.
        **distance***: the maximal Levenshtein distance for spelling
                       suggestions (default: 1, max: 4).
        **include**: specifies an inclusion custom dictionary.
        **exclude**: specifies an exclusion custom dictionary.

        For more information see `FT.SPELLCHECK <https://redis.io/commands/ft.spellcheck>`_.
        """  # noqa
        cmd = [SPELLCHECK_CMD, self.index_name, query]
        if distance:
            cmd.extend(["DISTANCE", distance])

        if include:
            cmd.extend(["TERMS", "INCLUDE", include])

        if exclude:
            cmd.extend(["TERMS", "EXCLUDE", exclude])

        raw = self.execute_command(*cmd)

        corrections = {}
        if raw == 0:
            return corrections

        for _correction in raw:
            if isinstance(_correction, int) and _correction == 0:
                continue

            if len(_correction) != 3:
                continue
            if not _correction[2]:
                continue
            if not _correction[2][0]:
                continue

            # For spellcheck output
            # 1)  1) "TERM"
            #     2) "{term1}"
            #     3)  1)  1)  "{score1}"
            #             2)  "{suggestion1}"
            #         2)  1)  "{score2}"
            #             2)  "{suggestion2}"
            #
            # Following dictionary will be made
            # corrections = {
            #     '{term1}': [
            #         {'score': '{score1}', 'suggestion': '{suggestion1}'},
            #         {'score': '{score2}', 'suggestion': '{suggestion2}'}
            #     ]
            # }
            corrections[_correction[1]] = [
                {"score": _item[0], "suggestion": _item[1]} for _item in _correction[2]
            ]

        return corrections

    def dict_add(self, name, *terms):
        """Adds terms to a dictionary.

        ### Parameters

        - **name**: Dictionary name.
        - **terms**: List of items for adding to the dictionary.

        For more information see `FT.DICTADD <https://redis.io/commands/ft.dictadd>`_.
        """  # noqa
        cmd = [DICT_ADD_CMD, name]
        cmd.extend(terms)
        return self.execute_command(*cmd)

    def dict_del(self, name, *terms):
        """Deletes terms from a dictionary.

        ### Parameters

        - **name**: Dictionary name.
        - **terms**: List of items for removing from the dictionary.

        For more information see `FT.DICTDEL <https://redis.io/commands/ft.dictdel>`_.
        """  # noqa
        cmd = [DICT_DEL_CMD, name]
        cmd.extend(terms)
        return self.execute_command(*cmd)

    def dict_dump(self, name):
        """Dumps all terms in the given dictionary.

        ### Parameters

        - **name**: Dictionary name.

        For more information see `FT.DICTDUMP <https://redis.io/commands/ft.dictdump>`_.
        """  # noqa
        cmd = [DICT_DUMP_CMD, name]
        return self.execute_command(*cmd)

    def config_set(self, option, value):
        """Set runtime configuration option.

        ### Parameters

        - **option**: the name of the configuration option.
        - **value**: a value for the configuration option.

        For more information see `FT.CONFIG SET <https://redis.io/commands/ft.config-set>`_.
        """  # noqa
        cmd = [CONFIG_CMD, "SET", option, value]
        raw = self.execute_command(*cmd)
        return raw == "OK"

    def config_get(self, option):
        """Get runtime configuration option value.

        ### Parameters

        - **option**: the name of the configuration option.

        For more information see `FT.CONFIG GET <https://redis.io/commands/ft.config-get>`_.
        """  # noqa
        cmd = [CONFIG_CMD, "GET", option]
        res = {}
        raw = self.execute_command(*cmd)
        if raw:
            for kvs in raw:
                res[kvs[0]] = kvs[1]
        return res

    def tagvals(self, tagfield):
        """
        Return a list of all possible tag values

        ### Parameters

        - **tagfield**: Tag field name

        For more information see `FT.TAGVALS <https://redis.io/commands/ft.tagvals>`_.
        """  # noqa

        return self.execute_command(TAGVALS_CMD, self.index_name, tagfield)

    def aliasadd(self, alias):
        """
        Alias a search index - will fail if alias already exists

        ### Parameters

        - **alias**: Name of the alias to create

        For more information see `FT.ALIASADD <https://redis.io/commands/ft.aliasadd>`_.
        """  # noqa

        return self.execute_command(ALIAS_ADD_CMD, alias, self.index_name)

    def aliasupdate(self, alias):
        """
        Updates an alias - will fail if alias does not already exist

        ### Parameters

        - **alias**: Name of the alias to create

        For more information see `FT.ALIASUPDATE <https://redis.io/commands/ft.aliasupdate>`_.
        """  # noqa

        return self.execute_command(ALIAS_UPDATE_CMD, alias, self.index_name)

    def aliasdel(self, alias):
        """
        Removes an alias to a search index

        ### Parameters

        - **alias**: Name of the alias to delete

        For more information see `FT.ALIASDEL <https://redis.io/commands/ft.aliasdel>`_.
        """  # noqa
        return self.execute_command(ALIAS_DEL_CMD, alias)

    def sugadd(self, key, *suggestions, **kwargs):
        """
        Add suggestion terms to the AutoCompleter engine. Each suggestion has
        a score and string.
        If kwargs["increment"] is true and the terms are already in the
        server's dictionary, we increment their scores.

        For more information see `FT.SUGADD <https://redis.io/commands/ft.sugadd/>`_.
        """  # noqa
        # If Transaction is not False it will MULTI/EXEC which will error
        pipe = self.pipeline(transaction=False)
        for sug in suggestions:
            args = [SUGADD_COMMAND, key, sug.string, sug.score]
            if kwargs.get("increment"):
                args.append("INCR")
            if sug.payload:
                args.append("PAYLOAD")
                args.append(sug.payload)

            pipe.execute_command(*args)

        return pipe.execute()[-1]

    def suglen(self, key):
        """
        Return the number of entries in the AutoCompleter index.

        For more information see `FT.SUGLEN <https://redis.io/commands/ft.suglen>`_.
        """  # noqa
        return self.execute_command(SUGLEN_COMMAND, key)

    def sugdel(self, key, string):
        """
        Delete a string from the AutoCompleter index.
        Returns 1 if the string was found and deleted, 0 otherwise.

        For more information see `FT.SUGDEL <https://redis.io/commands/ft.sugdel>`_.
        """  # noqa
        return self.execute_command(SUGDEL_COMMAND, key, string)

    def sugget(
        self, key, prefix, fuzzy=False, num=10, with_scores=False, with_payloads=False
    ):
        """
        Get a list of suggestions from the AutoCompleter, for a given prefix.

        Parameters:

        prefix : str
            The prefix we are searching. **Must be valid ascii or utf-8**
        fuzzy : bool
            If set to true, the prefix search is done in fuzzy mode.
            **NOTE**: Running fuzzy searches on short (<3 letters) prefixes
            can be very
            slow, and even scan the entire index.
        with_scores : bool
            If set to true, we also return the (refactored) score of
            each suggestion.
            This is normally not needed, and is NOT the original score
            inserted into the index.
        with_payloads : bool
            Return suggestion payloads
        num : int
            The maximum number of results we return. Note that we might
            return less. The algorithm trims irrelevant suggestions.

        Returns:

        list:
             A list of Suggestion objects. If with_scores was False, the
             score of all suggestions is 1.

        For more information see `FT.SUGGET <https://redis.io/commands/ft.sugget>`_.
        """  # noqa
        args = [SUGGET_COMMAND, key, prefix, "MAX", num]
        if fuzzy:
            args.append(FUZZY)
        if with_scores:
            args.append(WITHSCORES)
        if with_payloads:
            args.append(WITHPAYLOADS)

        ret = self.execute_command(*args)
        results = []
        if not ret:
            return results

        parser = SuggestionParser(with_scores, with_payloads, ret)
        return [s for s in parser]

    def synupdate(self, groupid, skipinitial=False, *terms):
        """
        Updates a synonym group.
        The command is used to create or update a synonym group with
        additional terms.
        Only documents which were indexed after the update will be affected.

        Parameters:

        groupid :
            Synonym group id.
        skipinitial : bool
            If set to true, we do not scan and index.
        terms :
            The terms.

        For more information see `FT.SYNUPDATE <https://redis.io/commands/ft.synupdate>`_.
        """  # noqa
        cmd = [SYNUPDATE_CMD, self.index_name, groupid]
        if skipinitial:
            cmd.extend(["SKIPINITIALSCAN"])
        cmd.extend(terms)
        return self.execute_command(*cmd)

    def syndump(self):
        """
        Dumps the contents of a synonym group.

        The command is used to dump the synonyms data structure.
        Returns a list of synonym terms and their synonym group ids.

        For more information see `FT.SYNDUMP <https://redis.io/commands/ft.syndump>`_.
        """  # noqa
        raw = self.execute_command(SYNDUMP_CMD, self.index_name)
        return {raw[i]: raw[i + 1] for i in range(0, len(raw), 2)}


class AsyncSearchCommands(SearchCommands):
    async def info(self):
        """
        Get info an stats about the the current index, including the number of
        documents, memory consumption, etc

        For more information see `FT.INFO <https://redis.io/commands/ft.info>`_.
        """

        res = await self.execute_command(INFO_CMD, self.index_name)
        it = map(to_string, res)
        return dict(zip(it, it))

    async def search(
        self,
        query: Union[str, Query],
        query_params: Dict[str, Union[str, int, float]] = None,
    ):
        """
        Search the index for a given query, and return a result of documents

        ### Parameters

        - **query**: the search query. Either a text for simple queries with
                     default parameters, or a Query object for complex queries.
                     See RediSearch's documentation on query format

        For more information see `FT.SEARCH <https://redis.io/commands/ft.search>`_.
        """  # noqa
        args, query = self._mk_query_args(query, query_params=query_params)
        st = time.time()
        res = await self.execute_command(SEARCH_CMD, *args)

        if isinstance(res, Pipeline):
            return res

        return Result(
            res,
            not query._no_content,
            duration=(time.time() - st) * 1000.0,
            has_payload=query._with_payloads,
            with_scores=query._with_scores,
        )

    async def aggregate(
        self,
        query: Union[str, Query],
        query_params: Dict[str, Union[str, int, float]] = None,
    ):
        """
        Issue an aggregation query.

        ### Parameters

        **query**: This can be either an `AggregateRequest`, or a `Cursor`

        An `AggregateResult` object is returned. You can access the rows from
        its `rows` property, which will always yield the rows of the result.

        For more information see `FT.AGGREGATE <https://redis.io/commands/ft.aggregate>`_.
        """  # noqa
        if isinstance(query, AggregateRequest):
            has_cursor = bool(query._cursor)
            cmd = [AGGREGATE_CMD, self.index_name] + query.build_args()
        elif isinstance(query, Cursor):
            has_cursor = True
            cmd = [CURSOR_CMD, "READ", self.index_name] + query.build_args()
        else:
            raise ValueError("Bad query", query)
        cmd += self.get_params_args(query_params)

        raw = await self.execute_command(*cmd)
        return self._get_aggregate_result(raw, query, has_cursor)

    async def spellcheck(self, query, distance=None, include=None, exclude=None):
        """
        Issue a spellcheck query

        ### Parameters

        **query**: search query.
        **distance***: the maximal Levenshtein distance for spelling
                       suggestions (default: 1, max: 4).
        **include**: specifies an inclusion custom dictionary.
        **exclude**: specifies an exclusion custom dictionary.

        For more information see `FT.SPELLCHECK <https://redis.io/commands/ft.spellcheck>`_.
        """  # noqa
        cmd = [SPELLCHECK_CMD, self.index_name, query]
        if distance:
            cmd.extend(["DISTANCE", distance])

        if include:
            cmd.extend(["TERMS", "INCLUDE", include])

        if exclude:
            cmd.extend(["TERMS", "EXCLUDE", exclude])

        raw = await self.execute_command(*cmd)

        corrections = {}
        if raw == 0:
            return corrections

        for _correction in raw:
            if isinstance(_correction, int) and _correction == 0:
                continue

            if len(_correction) != 3:
                continue
            if not _correction[2]:
                continue
            if not _correction[2][0]:
                continue

            corrections[_correction[1]] = [
                {"score": _item[0], "suggestion": _item[1]} for _item in _correction[2]
            ]

        return corrections

    async def config_set(self, option, value):
        """Set runtime configuration option.

        ### Parameters

        - **option**: the name of the configuration option.
        - **value**: a value for the configuration option.

        For more information see `FT.CONFIG SET <https://redis.io/commands/ft.config-set>`_.
        """  # noqa
        cmd = [CONFIG_CMD, "SET", option, value]
        raw = await self.execute_command(*cmd)
        return raw == "OK"

    async def config_get(self, option):
        """Get runtime configuration option value.

        ### Parameters

        - **option**: the name of the configuration option.

        For more information see `FT.CONFIG GET <https://redis.io/commands/ft.config-get>`_.
        """  # noqa
        cmd = [CONFIG_CMD, "GET", option]
        res = {}
        raw = await self.execute_command(*cmd)
        if raw:
            for kvs in raw:
                res[kvs[0]] = kvs[1]
        return res

    async def load_document(self, id):
        """
        Load a single document by id
        """
        fields = await self.client.hgetall(id)
        f2 = {to_string(k): to_string(v) for k, v in fields.items()}
        fields = f2

        try:
            del fields["id"]
        except KeyError:
            pass

        return Document(id=id, **fields)

    async def sugadd(self, key, *suggestions, **kwargs):
        """
        Add suggestion terms to the AutoCompleter engine. Each suggestion has
        a score and string.
        If kwargs["increment"] is true and the terms are already in the
        server's dictionary, we increment their scores.

        For more information see `FT.SUGADD <https://redis.io/commands/ft.sugadd>`_.
        """  # noqa
        # If Transaction is not False it will MULTI/EXEC which will error
        pipe = self.pipeline(transaction=False)
        for sug in suggestions:
            args = [SUGADD_COMMAND, key, sug.string, sug.score]
            if kwargs.get("increment"):
                args.append("INCR")
            if sug.payload:
                args.append("PAYLOAD")
                args.append(sug.payload)

            pipe.execute_command(*args)

        return (await pipe.execute())[-1]

    async def sugget(
        self, key, prefix, fuzzy=False, num=10, with_scores=False, with_payloads=False
    ):
        """
        Get a list of suggestions from the AutoCompleter, for a given prefix.

        Parameters:

        prefix : str
            The prefix we are searching. **Must be valid ascii or utf-8**
        fuzzy : bool
            If set to true, the prefix search is done in fuzzy mode.
            **NOTE**: Running fuzzy searches on short (<3 letters) prefixes
            can be very
            slow, and even scan the entire index.
        with_scores : bool
            If set to true, we also return the (refactored) score of
            each suggestion.
            This is normally not needed, and is NOT the original score
            inserted into the index.
        with_payloads : bool
            Return suggestion payloads
        num : int
            The maximum number of results we return. Note that we might
            return less. The algorithm trims irrelevant suggestions.

        Returns:

        list:
             A list of Suggestion objects. If with_scores was False, the
             score of all suggestions is 1.

        For more information see `FT.SUGGET <https://redis.io/commands/ft.sugget>`_.
        """  # noqa
        args = [SUGGET_COMMAND, key, prefix, "MAX", num]
        if fuzzy:
            args.append(FUZZY)
        if with_scores:
            args.append(WITHSCORES)
        if with_payloads:
            args.append(WITHPAYLOADS)

        ret = await self.execute_command(*args)
        results = []
        if not ret:
            return results

        parser = SuggestionParser(with_scores, with_payloads, ret)
        return [s for s in parser]
