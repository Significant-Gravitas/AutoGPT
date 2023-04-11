from redis.client import bool_ok

from ..helpers import parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo


class AbstractBloom(object):
    """
    The client allows to interact with RedisBloom and use all of
    it's functionality.

    - BF for Bloom Filter
    - CF for Cuckoo Filter
    - CMS for Count-Min Sketch
    - TOPK for TopK Data Structure
    - TDIGEST for estimate rank statistics
    """

    @staticmethod
    def append_items(params, items):
        """Append ITEMS to params."""
        params.extend(["ITEMS"])
        params += items

    @staticmethod
    def append_error(params, error):
        """Append ERROR to params."""
        if error is not None:
            params.extend(["ERROR", error])

    @staticmethod
    def append_capacity(params, capacity):
        """Append CAPACITY to params."""
        if capacity is not None:
            params.extend(["CAPACITY", capacity])

    @staticmethod
    def append_expansion(params, expansion):
        """Append EXPANSION to params."""
        if expansion is not None:
            params.extend(["EXPANSION", expansion])

    @staticmethod
    def append_no_scale(params, noScale):
        """Append NONSCALING tag to params."""
        if noScale is not None:
            params.extend(["NONSCALING"])

    @staticmethod
    def append_weights(params, weights):
        """Append WEIGHTS to params."""
        if len(weights) > 0:
            params.append("WEIGHTS")
            params += weights

    @staticmethod
    def append_no_create(params, noCreate):
        """Append NOCREATE tag to params."""
        if noCreate is not None:
            params.extend(["NOCREATE"])

    @staticmethod
    def append_items_and_increments(params, items, increments):
        """Append pairs of items and increments to params."""
        for i in range(len(items)):
            params.append(items[i])
            params.append(increments[i])

    @staticmethod
    def append_values_and_weights(params, items, weights):
        """Append pairs of items and weights to params."""
        for i in range(len(items)):
            params.append(items[i])
            params.append(weights[i])

    @staticmethod
    def append_max_iterations(params, max_iterations):
        """Append MAXITERATIONS to params."""
        if max_iterations is not None:
            params.extend(["MAXITERATIONS", max_iterations])

    @staticmethod
    def append_bucket_size(params, bucket_size):
        """Append BUCKETSIZE to params."""
        if bucket_size is not None:
            params.extend(["BUCKETSIZE", bucket_size])


class CMSBloom(CMSCommands, AbstractBloom):
    def __init__(self, client, **kwargs):
        """Create a new RedisBloom client."""
        # Set the module commands' callbacks
        MODULE_CALLBACKS = {
            CMS_INITBYDIM: bool_ok,
            CMS_INITBYPROB: bool_ok,
            # CMS_INCRBY: spaceHolder,
            # CMS_QUERY: spaceHolder,
            CMS_MERGE: bool_ok,
            CMS_INFO: CMSInfo,
        }

        self.client = client
        self.commandmixin = CMSCommands
        self.execute_command = client.execute_command

        for k, v in MODULE_CALLBACKS.items():
            self.client.set_response_callback(k, v)


class TOPKBloom(TOPKCommands, AbstractBloom):
    def __init__(self, client, **kwargs):
        """Create a new RedisBloom client."""
        # Set the module commands' callbacks
        MODULE_CALLBACKS = {
            TOPK_RESERVE: bool_ok,
            TOPK_ADD: parse_to_list,
            TOPK_INCRBY: parse_to_list,
            # TOPK_QUERY: spaceHolder,
            # TOPK_COUNT: spaceHolder,
            TOPK_LIST: parse_to_list,
            TOPK_INFO: TopKInfo,
        }

        self.client = client
        self.commandmixin = TOPKCommands
        self.execute_command = client.execute_command

        for k, v in MODULE_CALLBACKS.items():
            self.client.set_response_callback(k, v)


class CFBloom(CFCommands, AbstractBloom):
    def __init__(self, client, **kwargs):
        """Create a new RedisBloom client."""
        # Set the module commands' callbacks
        MODULE_CALLBACKS = {
            CF_RESERVE: bool_ok,
            # CF_ADD: spaceHolder,
            # CF_ADDNX: spaceHolder,
            # CF_INSERT: spaceHolder,
            # CF_INSERTNX: spaceHolder,
            # CF_EXISTS: spaceHolder,
            # CF_DEL: spaceHolder,
            # CF_COUNT: spaceHolder,
            # CF_SCANDUMP: spaceHolder,
            # CF_LOADCHUNK: spaceHolder,
            CF_INFO: CFInfo,
        }

        self.client = client
        self.commandmixin = CFCommands
        self.execute_command = client.execute_command

        for k, v in MODULE_CALLBACKS.items():
            self.client.set_response_callback(k, v)


class TDigestBloom(TDigestCommands, AbstractBloom):
    def __init__(self, client, **kwargs):
        """Create a new RedisBloom client."""
        # Set the module commands' callbacks
        MODULE_CALLBACKS = {
            TDIGEST_CREATE: bool_ok,
            # TDIGEST_RESET: bool_ok,
            # TDIGEST_ADD: spaceHolder,
            # TDIGEST_MERGE: spaceHolder,
            TDIGEST_CDF: parse_to_list,
            TDIGEST_QUANTILE: parse_to_list,
            TDIGEST_MIN: float,
            TDIGEST_MAX: float,
            TDIGEST_TRIMMED_MEAN: float,
            TDIGEST_INFO: TDigestInfo,
            TDIGEST_RANK: parse_to_list,
            TDIGEST_REVRANK: parse_to_list,
            TDIGEST_BYRANK: parse_to_list,
            TDIGEST_BYREVRANK: parse_to_list,
        }

        self.client = client
        self.commandmixin = TDigestCommands
        self.execute_command = client.execute_command

        for k, v in MODULE_CALLBACKS.items():
            self.client.set_response_callback(k, v)


class BFBloom(BFCommands, AbstractBloom):
    def __init__(self, client, **kwargs):
        """Create a new RedisBloom client."""
        # Set the module commands' callbacks
        MODULE_CALLBACKS = {
            BF_RESERVE: bool_ok,
            # BF_ADD: spaceHolder,
            # BF_MADD: spaceHolder,
            # BF_INSERT: spaceHolder,
            # BF_EXISTS: spaceHolder,
            # BF_MEXISTS: spaceHolder,
            # BF_SCANDUMP: spaceHolder,
            # BF_LOADCHUNK: spaceHolder,
            # BF_CARD: spaceHolder,
            BF_INFO: BFInfo,
        }

        self.client = client
        self.commandmixin = BFCommands
        self.execute_command = client.execute_command

        for k, v in MODULE_CALLBACKS.items():
            self.client.set_response_callback(k, v)
