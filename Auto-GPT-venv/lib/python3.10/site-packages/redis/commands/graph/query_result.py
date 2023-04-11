import sys
from collections import OrderedDict
from distutils.util import strtobool

# from prettytable import PrettyTable
from redis import ResponseError

from .edge import Edge
from .exceptions import VersionMismatchException
from .node import Node
from .path import Path

LABELS_ADDED = "Labels added"
LABELS_REMOVED = "Labels removed"
NODES_CREATED = "Nodes created"
NODES_DELETED = "Nodes deleted"
RELATIONSHIPS_DELETED = "Relationships deleted"
PROPERTIES_SET = "Properties set"
PROPERTIES_REMOVED = "Properties removed"
RELATIONSHIPS_CREATED = "Relationships created"
INDICES_CREATED = "Indices created"
INDICES_DELETED = "Indices deleted"
CACHED_EXECUTION = "Cached execution"
INTERNAL_EXECUTION_TIME = "internal execution time"

STATS = [
    LABELS_ADDED,
    LABELS_REMOVED,
    NODES_CREATED,
    PROPERTIES_SET,
    PROPERTIES_REMOVED,
    RELATIONSHIPS_CREATED,
    NODES_DELETED,
    RELATIONSHIPS_DELETED,
    INDICES_CREATED,
    INDICES_DELETED,
    CACHED_EXECUTION,
    INTERNAL_EXECUTION_TIME,
]


class ResultSetColumnTypes:
    COLUMN_UNKNOWN = 0
    COLUMN_SCALAR = 1
    COLUMN_NODE = 2  # Unused as of RedisGraph v2.1.0, retained for backwards compatibility. # noqa
    COLUMN_RELATION = 3  # Unused as of RedisGraph v2.1.0, retained for backwards compatibility. # noqa


class ResultSetScalarTypes:
    VALUE_UNKNOWN = 0
    VALUE_NULL = 1
    VALUE_STRING = 2
    VALUE_INTEGER = 3
    VALUE_BOOLEAN = 4
    VALUE_DOUBLE = 5
    VALUE_ARRAY = 6
    VALUE_EDGE = 7
    VALUE_NODE = 8
    VALUE_PATH = 9
    VALUE_MAP = 10
    VALUE_POINT = 11


class QueryResult:
    def __init__(self, graph, response, profile=False):
        """
        A class that represents a result of the query operation.

        Args:

        graph:
            The graph on which the query was executed.
        response:
            The response from the server.
        profile:
            A boolean indicating if the query command was "GRAPH.PROFILE"
        """
        self.graph = graph
        self.header = []
        self.result_set = []

        # in case of an error an exception will be raised
        self._check_for_errors(response)

        if len(response) == 1:
            self.parse_statistics(response[0])
        elif profile:
            self.parse_profile(response)
        else:
            # start by parsing statistics, matches the one we have
            self.parse_statistics(response[-1])  # Last element.
            self.parse_results(response)

    def _check_for_errors(self, response):
        """
        Check if the response contains an error.
        """
        if isinstance(response[0], ResponseError):
            error = response[0]
            if str(error) == "version mismatch":
                version = response[1]
                error = VersionMismatchException(version)
            raise error

        # If we encountered a run-time error, the last response
        # element will be an exception
        if isinstance(response[-1], ResponseError):
            raise response[-1]

    def parse_results(self, raw_result_set):
        """
        Parse the query execution result returned from the server.
        """
        self.header = self.parse_header(raw_result_set)

        # Empty header.
        if len(self.header) == 0:
            return

        self.result_set = self.parse_records(raw_result_set)

    def parse_statistics(self, raw_statistics):
        """
        Parse the statistics returned in the response.
        """
        self.statistics = {}

        # decode statistics
        for idx, stat in enumerate(raw_statistics):
            if isinstance(stat, bytes):
                raw_statistics[idx] = stat.decode()

        for s in STATS:
            v = self._get_value(s, raw_statistics)
            if v is not None:
                self.statistics[s] = v

    def parse_header(self, raw_result_set):
        """
        Parse the header of the result.
        """
        # An array of column name/column type pairs.
        header = raw_result_set[0]
        return header

    def parse_records(self, raw_result_set):
        """
        Parses the result set and returns a list of records.
        """
        records = [
            [
                self.parse_record_types[self.header[idx][0]](cell)
                for idx, cell in enumerate(row)
            ]
            for row in raw_result_set[1]
        ]

        return records

    def parse_entity_properties(self, props):
        """
        Parse node / edge properties.
        """
        # [[name, value type, value] X N]
        properties = {}
        for prop in props:
            prop_name = self.graph.get_property(prop[0])
            prop_value = self.parse_scalar(prop[1:])
            properties[prop_name] = prop_value

        return properties

    def parse_string(self, cell):
        """
        Parse the cell as a string.
        """
        if isinstance(cell, bytes):
            return cell.decode()
        elif not isinstance(cell, str):
            return str(cell)
        else:
            return cell

    def parse_node(self, cell):
        """
        Parse the cell to a node.
        """
        # Node ID (integer),
        # [label string offset (integer)],
        # [[name, value type, value] X N]

        node_id = int(cell[0])
        labels = None
        if len(cell[1]) > 0:
            labels = []
            for inner_label in cell[1]:
                labels.append(self.graph.get_label(inner_label))
        properties = self.parse_entity_properties(cell[2])
        return Node(node_id=node_id, label=labels, properties=properties)

    def parse_edge(self, cell):
        """
        Parse the cell to an edge.
        """
        # Edge ID (integer),
        # reltype string offset (integer),
        # src node ID offset (integer),
        # dest node ID offset (integer),
        # [[name, value, value type] X N]

        edge_id = int(cell[0])
        relation = self.graph.get_relation(cell[1])
        src_node_id = int(cell[2])
        dest_node_id = int(cell[3])
        properties = self.parse_entity_properties(cell[4])
        return Edge(
            src_node_id, relation, dest_node_id, edge_id=edge_id, properties=properties
        )

    def parse_path(self, cell):
        """
        Parse the cell to a path.
        """
        nodes = self.parse_scalar(cell[0])
        edges = self.parse_scalar(cell[1])
        return Path(nodes, edges)

    def parse_map(self, cell):
        """
        Parse the cell as a map.
        """
        m = OrderedDict()
        n_entries = len(cell)

        # A map is an array of key value pairs.
        # 1. key (string)
        # 2. array: (value type, value)
        for i in range(0, n_entries, 2):
            key = self.parse_string(cell[i])
            m[key] = self.parse_scalar(cell[i + 1])

        return m

    def parse_point(self, cell):
        """
        Parse the cell to point.
        """
        p = {}
        # A point is received an array of the form: [latitude, longitude]
        # It is returned as a map of the form: {"latitude": latitude, "longitude": longitude} # noqa
        p["latitude"] = float(cell[0])
        p["longitude"] = float(cell[1])
        return p

    def parse_null(self, cell):
        """
        Parse a null value.
        """
        return None

    def parse_integer(self, cell):
        """
        Parse the integer value from the cell.
        """
        return int(cell)

    def parse_boolean(self, value):
        """
        Parse the cell value as a boolean.
        """
        value = value.decode() if isinstance(value, bytes) else value
        try:
            scalar = True if strtobool(value) else False
        except ValueError:
            sys.stderr.write("unknown boolean type\n")
            scalar = None
        return scalar

    def parse_double(self, cell):
        """
        Parse the cell as a double.
        """
        return float(cell)

    def parse_array(self, value):
        """
        Parse an array of values.
        """
        scalar = [self.parse_scalar(value[i]) for i in range(len(value))]
        return scalar

    def parse_unknown(self, cell):
        """
        Parse a cell of unknown type.
        """
        sys.stderr.write("Unknown type\n")
        return None

    def parse_scalar(self, cell):
        """
        Parse a scalar value from a cell in the result set.
        """
        scalar_type = int(cell[0])
        value = cell[1]
        scalar = self.parse_scalar_types[scalar_type](value)

        return scalar

    def parse_profile(self, response):
        self.result_set = [x[0 : x.index(",")].strip() for x in response]

    def is_empty(self):
        return len(self.result_set) == 0

    @staticmethod
    def _get_value(prop, statistics):
        for stat in statistics:
            if prop in stat:
                return float(stat.split(": ")[1].split(" ")[0])

        return None

    def _get_stat(self, stat):
        return self.statistics[stat] if stat in self.statistics else 0

    @property
    def labels_added(self):
        """Returns the number of labels added in the query"""
        return self._get_stat(LABELS_ADDED)

    @property
    def labels_removed(self):
        """Returns the number of labels removed in the query"""
        return self._get_stat(LABELS_REMOVED)

    @property
    def nodes_created(self):
        """Returns the number of nodes created in the query"""
        return self._get_stat(NODES_CREATED)

    @property
    def nodes_deleted(self):
        """Returns the number of nodes deleted in the query"""
        return self._get_stat(NODES_DELETED)

    @property
    def properties_set(self):
        """Returns the number of properties set in the query"""
        return self._get_stat(PROPERTIES_SET)

    @property
    def properties_removed(self):
        """Returns the number of properties removed in the query"""
        return self._get_stat(PROPERTIES_REMOVED)

    @property
    def relationships_created(self):
        """Returns the number of relationships created in the query"""
        return self._get_stat(RELATIONSHIPS_CREATED)

    @property
    def relationships_deleted(self):
        """Returns the number of relationships deleted in the query"""
        return self._get_stat(RELATIONSHIPS_DELETED)

    @property
    def indices_created(self):
        """Returns the number of indices created in the query"""
        return self._get_stat(INDICES_CREATED)

    @property
    def indices_deleted(self):
        """Returns the number of indices deleted in the query"""
        return self._get_stat(INDICES_DELETED)

    @property
    def cached_execution(self):
        """Returns whether or not the query execution plan was cached"""
        return self._get_stat(CACHED_EXECUTION) == 1

    @property
    def run_time_ms(self):
        """Returns the server execution time of the query"""
        return self._get_stat(INTERNAL_EXECUTION_TIME)

    @property
    def parse_scalar_types(self):
        return {
            ResultSetScalarTypes.VALUE_NULL: self.parse_null,
            ResultSetScalarTypes.VALUE_STRING: self.parse_string,
            ResultSetScalarTypes.VALUE_INTEGER: self.parse_integer,
            ResultSetScalarTypes.VALUE_BOOLEAN: self.parse_boolean,
            ResultSetScalarTypes.VALUE_DOUBLE: self.parse_double,
            ResultSetScalarTypes.VALUE_ARRAY: self.parse_array,
            ResultSetScalarTypes.VALUE_NODE: self.parse_node,
            ResultSetScalarTypes.VALUE_EDGE: self.parse_edge,
            ResultSetScalarTypes.VALUE_PATH: self.parse_path,
            ResultSetScalarTypes.VALUE_MAP: self.parse_map,
            ResultSetScalarTypes.VALUE_POINT: self.parse_point,
            ResultSetScalarTypes.VALUE_UNKNOWN: self.parse_unknown,
        }

    @property
    def parse_record_types(self):
        return {
            ResultSetColumnTypes.COLUMN_SCALAR: self.parse_scalar,
            ResultSetColumnTypes.COLUMN_NODE: self.parse_node,
            ResultSetColumnTypes.COLUMN_RELATION: self.parse_edge,
            ResultSetColumnTypes.COLUMN_UNKNOWN: self.parse_unknown,
        }


class AsyncQueryResult(QueryResult):
    """
    Async version for the QueryResult class - a class that
    represents a result of the query operation.
    """

    def __init__(self):
        """
        To init the class you must call self.initialize()
        """
        pass

    async def initialize(self, graph, response, profile=False):
        """
        Initializes the class.
        Args:

        graph:
            The graph on which the query was executed.
        response:
            The response from the server.
        profile:
            A boolean indicating if the query command was "GRAPH.PROFILE"
        """
        self.graph = graph
        self.header = []
        self.result_set = []

        # in case of an error an exception will be raised
        self._check_for_errors(response)

        if len(response) == 1:
            self.parse_statistics(response[0])
        elif profile:
            self.parse_profile(response)
        else:
            # start by parsing statistics, matches the one we have
            self.parse_statistics(response[-1])  # Last element.
            await self.parse_results(response)

        return self

    async def parse_node(self, cell):
        """
        Parses a node from the cell.
        """
        # Node ID (integer),
        # [label string offset (integer)],
        # [[name, value type, value] X N]

        labels = None
        if len(cell[1]) > 0:
            labels = []
            for inner_label in cell[1]:
                labels.append(await self.graph.get_label(inner_label))
        properties = await self.parse_entity_properties(cell[2])
        node_id = int(cell[0])
        return Node(node_id=node_id, label=labels, properties=properties)

    async def parse_scalar(self, cell):
        """
        Parses a scalar value from the server response.
        """
        scalar_type = int(cell[0])
        value = cell[1]
        try:
            scalar = await self.parse_scalar_types[scalar_type](value)
        except TypeError:
            # Not all of the functions are async
            scalar = self.parse_scalar_types[scalar_type](value)

        return scalar

    async def parse_records(self, raw_result_set):
        """
        Parses the result set and returns a list of records.
        """
        records = []
        for row in raw_result_set[1]:
            record = [
                await self.parse_record_types[self.header[idx][0]](cell)
                for idx, cell in enumerate(row)
            ]
            records.append(record)

        return records

    async def parse_results(self, raw_result_set):
        """
        Parse the query execution result returned from the server.
        """
        self.header = self.parse_header(raw_result_set)

        # Empty header.
        if len(self.header) == 0:
            return

        self.result_set = await self.parse_records(raw_result_set)

    async def parse_entity_properties(self, props):
        """
        Parse node / edge properties.
        """
        # [[name, value type, value] X N]
        properties = {}
        for prop in props:
            prop_name = await self.graph.get_property(prop[0])
            prop_value = await self.parse_scalar(prop[1:])
            properties[prop_name] = prop_value

        return properties

    async def parse_edge(self, cell):
        """
        Parse the cell to an edge.
        """
        # Edge ID (integer),
        # reltype string offset (integer),
        # src node ID offset (integer),
        # dest node ID offset (integer),
        # [[name, value, value type] X N]

        edge_id = int(cell[0])
        relation = await self.graph.get_relation(cell[1])
        src_node_id = int(cell[2])
        dest_node_id = int(cell[3])
        properties = await self.parse_entity_properties(cell[4])
        return Edge(
            src_node_id, relation, dest_node_id, edge_id=edge_id, properties=properties
        )

    async def parse_path(self, cell):
        """
        Parse the cell to a path.
        """
        nodes = await self.parse_scalar(cell[0])
        edges = await self.parse_scalar(cell[1])
        return Path(nodes, edges)

    async def parse_map(self, cell):
        """
        Parse the cell to a map.
        """
        m = OrderedDict()
        n_entries = len(cell)

        # A map is an array of key value pairs.
        # 1. key (string)
        # 2. array: (value type, value)
        for i in range(0, n_entries, 2):
            key = self.parse_string(cell[i])
            m[key] = await self.parse_scalar(cell[i + 1])

        return m

    async def parse_array(self, value):
        """
        Parse array value.
        """
        scalar = [await self.parse_scalar(value[i]) for i in range(len(value))]
        return scalar
