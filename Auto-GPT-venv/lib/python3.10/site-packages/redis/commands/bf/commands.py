from redis.client import NEVER_DECODE
from redis.exceptions import ModuleError
from redis.utils import HIREDIS_AVAILABLE, deprecated_function

BF_RESERVE = "BF.RESERVE"
BF_ADD = "BF.ADD"
BF_MADD = "BF.MADD"
BF_INSERT = "BF.INSERT"
BF_EXISTS = "BF.EXISTS"
BF_MEXISTS = "BF.MEXISTS"
BF_SCANDUMP = "BF.SCANDUMP"
BF_LOADCHUNK = "BF.LOADCHUNK"
BF_INFO = "BF.INFO"
BF_CARD = "BF.CARD"

CF_RESERVE = "CF.RESERVE"
CF_ADD = "CF.ADD"
CF_ADDNX = "CF.ADDNX"
CF_INSERT = "CF.INSERT"
CF_INSERTNX = "CF.INSERTNX"
CF_EXISTS = "CF.EXISTS"
CF_MEXISTS = "CF.MEXISTS"
CF_DEL = "CF.DEL"
CF_COUNT = "CF.COUNT"
CF_SCANDUMP = "CF.SCANDUMP"
CF_LOADCHUNK = "CF.LOADCHUNK"
CF_INFO = "CF.INFO"

CMS_INITBYDIM = "CMS.INITBYDIM"
CMS_INITBYPROB = "CMS.INITBYPROB"
CMS_INCRBY = "CMS.INCRBY"
CMS_QUERY = "CMS.QUERY"
CMS_MERGE = "CMS.MERGE"
CMS_INFO = "CMS.INFO"

TOPK_RESERVE = "TOPK.RESERVE"
TOPK_ADD = "TOPK.ADD"
TOPK_INCRBY = "TOPK.INCRBY"
TOPK_QUERY = "TOPK.QUERY"
TOPK_COUNT = "TOPK.COUNT"
TOPK_LIST = "TOPK.LIST"
TOPK_INFO = "TOPK.INFO"

TDIGEST_CREATE = "TDIGEST.CREATE"
TDIGEST_RESET = "TDIGEST.RESET"
TDIGEST_ADD = "TDIGEST.ADD"
TDIGEST_MERGE = "TDIGEST.MERGE"
TDIGEST_CDF = "TDIGEST.CDF"
TDIGEST_QUANTILE = "TDIGEST.QUANTILE"
TDIGEST_MIN = "TDIGEST.MIN"
TDIGEST_MAX = "TDIGEST.MAX"
TDIGEST_INFO = "TDIGEST.INFO"
TDIGEST_TRIMMED_MEAN = "TDIGEST.TRIMMED_MEAN"
TDIGEST_RANK = "TDIGEST.RANK"
TDIGEST_REVRANK = "TDIGEST.REVRANK"
TDIGEST_BYRANK = "TDIGEST.BYRANK"
TDIGEST_BYREVRANK = "TDIGEST.BYREVRANK"


class BFCommands:
    """Bloom Filter commands."""

    # region Bloom Filter Functions
    def create(self, key, errorRate, capacity, expansion=None, noScale=None):
        """
        Create a new Bloom Filter `key` with desired probability of false positives
        `errorRate` expected entries to be inserted as `capacity`.
        Default expansion value is 2. By default, filter is auto-scaling.
        For more information see `BF.RESERVE <https://redis.io/commands/bf.reserve>`_.
        """  # noqa
        params = [key, errorRate, capacity]
        self.append_expansion(params, expansion)
        self.append_no_scale(params, noScale)
        return self.execute_command(BF_RESERVE, *params)

    reserve = create

    def add(self, key, item):
        """
        Add to a Bloom Filter `key` an `item`.
        For more information see `BF.ADD <https://redis.io/commands/bf.add>`_.
        """  # noqa
        return self.execute_command(BF_ADD, key, item)

    def madd(self, key, *items):
        """
        Add to a Bloom Filter `key` multiple `items`.
        For more information see `BF.MADD <https://redis.io/commands/bf.madd>`_.
        """  # noqa
        return self.execute_command(BF_MADD, key, *items)

    def insert(
        self,
        key,
        items,
        capacity=None,
        error=None,
        noCreate=None,
        expansion=None,
        noScale=None,
    ):
        """
        Add to a Bloom Filter `key` multiple `items`.

        If `nocreate` remain `None` and `key` does not exist, a new Bloom Filter
        `key` will be created with desired probability of false positives `errorRate`
        and expected entries to be inserted as `size`.
        For more information see `BF.INSERT <https://redis.io/commands/bf.insert>`_.
        """  # noqa
        params = [key]
        self.append_capacity(params, capacity)
        self.append_error(params, error)
        self.append_expansion(params, expansion)
        self.append_no_create(params, noCreate)
        self.append_no_scale(params, noScale)
        self.append_items(params, items)

        return self.execute_command(BF_INSERT, *params)

    def exists(self, key, item):
        """
        Check whether an `item` exists in Bloom Filter `key`.
        For more information see `BF.EXISTS <https://redis.io/commands/bf.exists>`_.
        """  # noqa
        return self.execute_command(BF_EXISTS, key, item)

    def mexists(self, key, *items):
        """
        Check whether `items` exist in Bloom Filter `key`.
        For more information see `BF.MEXISTS <https://redis.io/commands/bf.mexists>`_.
        """  # noqa
        return self.execute_command(BF_MEXISTS, key, *items)

    def scandump(self, key, iter):
        """
        Begin an incremental save of the bloom filter `key`.

        This is useful for large bloom filters which cannot fit into the normal SAVE and RESTORE model.
        The first time this command is called, the value of `iter` should be 0.
        This command will return successive (iter, data) pairs until (0, NULL) to indicate completion.
        For more information see `BF.SCANDUMP <https://redis.io/commands/bf.scandump>`_.
        """  # noqa
        if HIREDIS_AVAILABLE:
            raise ModuleError("This command cannot be used when hiredis is available.")

        params = [key, iter]
        options = {}
        options[NEVER_DECODE] = []
        return self.execute_command(BF_SCANDUMP, *params, **options)

    def loadchunk(self, key, iter, data):
        """
        Restore a filter previously saved using SCANDUMP.

        See the SCANDUMP command for example usage.
        This command will overwrite any bloom filter stored under key.
        Ensure that the bloom filter will not be modified between invocations.
        For more information see `BF.LOADCHUNK <https://redis.io/commands/bf.loadchunk>`_.
        """  # noqa
        return self.execute_command(BF_LOADCHUNK, key, iter, data)

    def info(self, key):
        """
        Return capacity, size, number of filters, number of items inserted, and expansion rate.
        For more information see `BF.INFO <https://redis.io/commands/bf.info>`_.
        """  # noqa
        return self.execute_command(BF_INFO, key)

    def card(self, key):
        """
        Returns the cardinality of a Bloom filter - number of items that were added to a Bloom filter and detected as unique
        (items that caused at least one bit to be set in at least one sub-filter).
        For more information see `BF.CARD <https://redis.io/commands/bf.card>`_.
        """  # noqa
        return self.execute_command(BF_CARD, key)


class CFCommands:
    """Cuckoo Filter commands."""

    # region Cuckoo Filter Functions
    def create(
        self, key, capacity, expansion=None, bucket_size=None, max_iterations=None
    ):
        """
        Create a new Cuckoo Filter `key` an initial `capacity` items.
        For more information see `CF.RESERVE <https://redis.io/commands/cf.reserve>`_.
        """  # noqa
        params = [key, capacity]
        self.append_expansion(params, expansion)
        self.append_bucket_size(params, bucket_size)
        self.append_max_iterations(params, max_iterations)
        return self.execute_command(CF_RESERVE, *params)

    reserve = create

    def add(self, key, item):
        """
        Add an `item` to a Cuckoo Filter `key`.
        For more information see `CF.ADD <https://redis.io/commands/cf.add>`_.
        """  # noqa
        return self.execute_command(CF_ADD, key, item)

    def addnx(self, key, item):
        """
        Add an `item` to a Cuckoo Filter `key` only if item does not yet exist.
        Command might be slower that `add`.
        For more information see `CF.ADDNX <https://redis.io/commands/cf.addnx>`_.
        """  # noqa
        return self.execute_command(CF_ADDNX, key, item)

    def insert(self, key, items, capacity=None, nocreate=None):
        """
        Add multiple `items` to a Cuckoo Filter `key`, allowing the filter
        to be created with a custom `capacity` if it does not yet exist.
        `items` must be provided as a list.
        For more information see `CF.INSERT <https://redis.io/commands/cf.insert>`_.
        """  # noqa
        params = [key]
        self.append_capacity(params, capacity)
        self.append_no_create(params, nocreate)
        self.append_items(params, items)
        return self.execute_command(CF_INSERT, *params)

    def insertnx(self, key, items, capacity=None, nocreate=None):
        """
        Add multiple `items` to a Cuckoo Filter `key` only if they do not exist yet,
        allowing the filter to be created with a custom `capacity` if it does not yet exist.
        `items` must be provided as a list.
        For more information see `CF.INSERTNX <https://redis.io/commands/cf.insertnx>`_.
        """  # noqa
        params = [key]
        self.append_capacity(params, capacity)
        self.append_no_create(params, nocreate)
        self.append_items(params, items)
        return self.execute_command(CF_INSERTNX, *params)

    def exists(self, key, item):
        """
        Check whether an `item` exists in Cuckoo Filter `key`.
        For more information see `CF.EXISTS <https://redis.io/commands/cf.exists>`_.
        """  # noqa
        return self.execute_command(CF_EXISTS, key, item)

    def mexists(self, key, *items):
        """
        Check whether an `items` exist in Cuckoo Filter `key`.
        For more information see `CF.MEXISTS <https://redis.io/commands/cf.mexists>`_.
        """  # noqa
        return self.execute_command(CF_MEXISTS, key, *items)

    def delete(self, key, item):
        """
        Delete `item` from `key`.
        For more information see `CF.DEL <https://redis.io/commands/cf.del>`_.
        """  # noqa
        return self.execute_command(CF_DEL, key, item)

    def count(self, key, item):
        """
        Return the number of times an `item` may be in the `key`.
        For more information see `CF.COUNT <https://redis.io/commands/cf.count>`_.
        """  # noqa
        return self.execute_command(CF_COUNT, key, item)

    def scandump(self, key, iter):
        """
        Begin an incremental save of the Cuckoo filter `key`.

        This is useful for large Cuckoo filters which cannot fit into the normal
        SAVE and RESTORE model.
        The first time this command is called, the value of `iter` should be 0.
        This command will return successive (iter, data) pairs until
        (0, NULL) to indicate completion.
        For more information see `CF.SCANDUMP <https://redis.io/commands/cf.scandump>`_.
        """  # noqa
        return self.execute_command(CF_SCANDUMP, key, iter)

    def loadchunk(self, key, iter, data):
        """
        Restore a filter previously saved using SCANDUMP. See the SCANDUMP command for example usage.

        This command will overwrite any Cuckoo filter stored under key.
        Ensure that the Cuckoo filter will not be modified between invocations.
        For more information see `CF.LOADCHUNK <https://redis.io/commands/cf.loadchunk>`_.
        """  # noqa
        return self.execute_command(CF_LOADCHUNK, key, iter, data)

    def info(self, key):
        """
        Return size, number of buckets, number of filter, number of items inserted,
        number of items deleted, bucket size, expansion rate, and max iteration.
        For more information see `CF.INFO <https://redis.io/commands/cf.info>`_.
        """  # noqa
        return self.execute_command(CF_INFO, key)


class TOPKCommands:
    """TOP-k Filter commands."""

    def reserve(self, key, k, width, depth, decay):
        """
        Create a new Top-K Filter `key` with desired probability of false
        positives `errorRate` expected entries to be inserted as `size`.
        For more information see `TOPK.RESERVE <https://redis.io/commands/topk.reserve>`_.
        """  # noqa
        return self.execute_command(TOPK_RESERVE, key, k, width, depth, decay)

    def add(self, key, *items):
        """
        Add one `item` or more to a Top-K Filter `key`.
        For more information see `TOPK.ADD <https://redis.io/commands/topk.add>`_.
        """  # noqa
        return self.execute_command(TOPK_ADD, key, *items)

    def incrby(self, key, items, increments):
        """
        Add/increase `items` to a Top-K Sketch `key` by ''increments''.
        Both `items` and `increments` are lists.
        For more information see `TOPK.INCRBY <https://redis.io/commands/topk.incrby>`_.

        Example:

        >>> topkincrby('A', ['foo'], [1])
        """  # noqa
        params = [key]
        self.append_items_and_increments(params, items, increments)
        return self.execute_command(TOPK_INCRBY, *params)

    def query(self, key, *items):
        """
        Check whether one `item` or more is a Top-K item at `key`.
        For more information see `TOPK.QUERY <https://redis.io/commands/topk.query>`_.
        """  # noqa
        return self.execute_command(TOPK_QUERY, key, *items)

    @deprecated_function(version="4.4.0", reason="deprecated since redisbloom 2.4.0")
    def count(self, key, *items):
        """
        Return count for one `item` or more from `key`.
        For more information see `TOPK.COUNT <https://redis.io/commands/topk.count>`_.
        """  # noqa
        return self.execute_command(TOPK_COUNT, key, *items)

    def list(self, key, withcount=False):
        """
        Return full list of items in Top-K list of `key`.
        If `withcount` set to True, return full list of items
        with probabilistic count in Top-K list of `key`.
        For more information see `TOPK.LIST <https://redis.io/commands/topk.list>`_.
        """  # noqa
        params = [key]
        if withcount:
            params.append("WITHCOUNT")
        return self.execute_command(TOPK_LIST, *params)

    def info(self, key):
        """
        Return k, width, depth and decay values of `key`.
        For more information see `TOPK.INFO <https://redis.io/commands/topk.info>`_.
        """  # noqa
        return self.execute_command(TOPK_INFO, key)


class TDigestCommands:
    def create(self, key, compression=100):
        """
        Allocate the memory and initialize the t-digest.
        For more information see `TDIGEST.CREATE <https://redis.io/commands/tdigest.create>`_.
        """  # noqa
        return self.execute_command(TDIGEST_CREATE, key, "COMPRESSION", compression)

    def reset(self, key):
        """
        Reset the sketch `key` to zero - empty out the sketch and re-initialize it.
        For more information see `TDIGEST.RESET <https://redis.io/commands/tdigest.reset>`_.
        """  # noqa
        return self.execute_command(TDIGEST_RESET, key)

    def add(self, key, values):
        """
        Adds one or more observations to a t-digest sketch `key`.

        For more information see `TDIGEST.ADD <https://redis.io/commands/tdigest.add>`_.
        """  # noqa
        return self.execute_command(TDIGEST_ADD, key, *values)

    def merge(self, destination_key, num_keys, *keys, compression=None, override=False):
        """
        Merges all of the values from `keys` to 'destination-key' sketch.
        It is mandatory to provide the `num_keys` before passing the input keys and
        the other (optional) arguments.
        If `destination_key` already exists its values are merged with the input keys.
        If you wish to override the destination key contents use the `OVERRIDE` parameter.

        For more information see `TDIGEST.MERGE <https://redis.io/commands/tdigest.merge>`_.
        """  # noqa
        params = [destination_key, num_keys, *keys]
        if compression is not None:
            params.extend(["COMPRESSION", compression])
        if override:
            params.append("OVERRIDE")
        return self.execute_command(TDIGEST_MERGE, *params)

    def min(self, key):
        """
        Return minimum value from the sketch `key`. Will return DBL_MAX if the sketch is empty.
        For more information see `TDIGEST.MIN <https://redis.io/commands/tdigest.min>`_.
        """  # noqa
        return self.execute_command(TDIGEST_MIN, key)

    def max(self, key):
        """
        Return maximum value from the sketch `key`. Will return DBL_MIN if the sketch is empty.
        For more information see `TDIGEST.MAX <https://redis.io/commands/tdigest.max>`_.
        """  # noqa
        return self.execute_command(TDIGEST_MAX, key)

    def quantile(self, key, quantile, *quantiles):
        """
        Returns estimates of one or more cutoffs such that a specified fraction of the
        observations added to this t-digest would be less than or equal to each of the
        specified cutoffs. (Multiple quantiles can be returned with one call)
        For more information see `TDIGEST.QUANTILE <https://redis.io/commands/tdigest.quantile>`_.
        """  # noqa
        return self.execute_command(TDIGEST_QUANTILE, key, quantile, *quantiles)

    def cdf(self, key, value, *values):
        """
        Return double fraction of all points added which are <= value.
        For more information see `TDIGEST.CDF <https://redis.io/commands/tdigest.cdf>`_.
        """  # noqa
        return self.execute_command(TDIGEST_CDF, key, value, *values)

    def info(self, key):
        """
        Return Compression, Capacity, Merged Nodes, Unmerged Nodes, Merged Weight, Unmerged Weight
        and Total Compressions.
        For more information see `TDIGEST.INFO <https://redis.io/commands/tdigest.info>`_.
        """  # noqa
        return self.execute_command(TDIGEST_INFO, key)

    def trimmed_mean(self, key, low_cut_quantile, high_cut_quantile):
        """
        Return mean value from the sketch, excluding observation values outside
        the low and high cutoff quantiles.
        For more information see `TDIGEST.TRIMMED_MEAN <https://redis.io/commands/tdigest.trimmed_mean>`_.
        """  # noqa
        return self.execute_command(
            TDIGEST_TRIMMED_MEAN, key, low_cut_quantile, high_cut_quantile
        )

    def rank(self, key, value, *values):
        """
        Retrieve the estimated rank of value (the number of observations in the sketch
        that are smaller than value + half the number of observations that are equal to value).

        For more information see `TDIGEST.RANK <https://redis.io/commands/tdigest.rank>`_.
        """  # noqa
        return self.execute_command(TDIGEST_RANK, key, value, *values)

    def revrank(self, key, value, *values):
        """
        Retrieve the estimated rank of value (the number of observations in the sketch
        that are larger than value + half the number of observations that are equal to value).

        For more information see `TDIGEST.REVRANK <https://redis.io/commands/tdigest.revrank>`_.
        """  # noqa
        return self.execute_command(TDIGEST_REVRANK, key, value, *values)

    def byrank(self, key, rank, *ranks):
        """
        Retrieve an estimation of the value with the given rank.

        For more information see `TDIGEST.BY_RANK <https://redis.io/commands/tdigest.by_rank>`_.
        """  # noqa
        return self.execute_command(TDIGEST_BYRANK, key, rank, *ranks)

    def byrevrank(self, key, rank, *ranks):
        """
        Retrieve an estimation of the value with the given reverse rank.

        For more information see `TDIGEST.BY_REVRANK <https://redis.io/commands/tdigest.by_revrank>`_.
        """  # noqa
        return self.execute_command(TDIGEST_BYREVRANK, key, rank, *ranks)


class CMSCommands:
    """Count-Min Sketch Commands"""

    # region Count-Min Sketch Functions
    def initbydim(self, key, width, depth):
        """
        Initialize a Count-Min Sketch `key` to dimensions (`width`, `depth`) specified by user.
        For more information see `CMS.INITBYDIM <https://redis.io/commands/cms.initbydim>`_.
        """  # noqa
        return self.execute_command(CMS_INITBYDIM, key, width, depth)

    def initbyprob(self, key, error, probability):
        """
        Initialize a Count-Min Sketch `key` to characteristics (`error`, `probability`) specified by user.
        For more information see `CMS.INITBYPROB <https://redis.io/commands/cms.initbyprob>`_.
        """  # noqa
        return self.execute_command(CMS_INITBYPROB, key, error, probability)

    def incrby(self, key, items, increments):
        """
        Add/increase `items` to a Count-Min Sketch `key` by ''increments''.
        Both `items` and `increments` are lists.
        For more information see `CMS.INCRBY <https://redis.io/commands/cms.incrby>`_.

        Example:

        >>> cmsincrby('A', ['foo'], [1])
        """  # noqa
        params = [key]
        self.append_items_and_increments(params, items, increments)
        return self.execute_command(CMS_INCRBY, *params)

    def query(self, key, *items):
        """
        Return count for an `item` from `key`. Multiple items can be queried with one call.
        For more information see `CMS.QUERY <https://redis.io/commands/cms.query>`_.
        """  # noqa
        return self.execute_command(CMS_QUERY, key, *items)

    def merge(self, destKey, numKeys, srcKeys, weights=[]):
        """
        Merge `numKeys` of sketches into `destKey`. Sketches specified in `srcKeys`.
        All sketches must have identical width and depth.
        `Weights` can be used to multiply certain sketches. Default weight is 1.
        Both `srcKeys` and `weights` are lists.
        For more information see `CMS.MERGE <https://redis.io/commands/cms.merge>`_.
        """  # noqa
        params = [destKey, numKeys]
        params += srcKeys
        self.append_weights(params, weights)
        return self.execute_command(CMS_MERGE, *params)

    def info(self, key):
        """
        Return width, depth and total count of the sketch.
        For more information see `CMS.INFO <https://redis.io/commands/cms.info>`_.
        """  # noqa
        return self.execute_command(CMS_INFO, key)
