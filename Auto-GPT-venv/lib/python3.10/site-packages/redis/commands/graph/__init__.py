from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa

DB_LABELS = "DB.LABELS"
DB_RAELATIONSHIPTYPES = "DB.RELATIONSHIPTYPES"
DB_PROPERTYKEYS = "DB.PROPERTYKEYS"


class Graph(GraphCommands):
    """
    Graph, collection of nodes and edges.
    """

    def __init__(self, client, name=random_string()):
        """
        Create a new graph.
        """
        self.NAME = name  # Graph key
        self.client = client
        self.execute_command = client.execute_command

        self.nodes = {}
        self.edges = []
        self._labels = []  # List of node labels.
        self._properties = []  # List of properties.
        self._relationship_types = []  # List of relation types.
        self.version = 0  # Graph version

    @property
    def name(self):
        return self.NAME

    def _clear_schema(self):
        self._labels = []
        self._properties = []
        self._relationship_types = []

    def _refresh_schema(self):
        self._clear_schema()
        self._refresh_labels()
        self._refresh_relations()
        self._refresh_attributes()

    def _refresh_labels(self):
        lbls = self.labels()

        # Unpack data.
        self._labels = [l[0] for _, l in enumerate(lbls)]

    def _refresh_relations(self):
        rels = self.relationship_types()

        # Unpack data.
        self._relationship_types = [r[0] for _, r in enumerate(rels)]

    def _refresh_attributes(self):
        props = self.property_keys()

        # Unpack data.
        self._properties = [p[0] for _, p in enumerate(props)]

    def get_label(self, idx):
        """
        Returns a label by it's index

        Args:

        idx:
            The index of the label
        """
        try:
            label = self._labels[idx]
        except IndexError:
            # Refresh labels.
            self._refresh_labels()
            label = self._labels[idx]
        return label

    def get_relation(self, idx):
        """
        Returns a relationship type by it's index

        Args:

        idx:
            The index of the relation
        """
        try:
            relationship_type = self._relationship_types[idx]
        except IndexError:
            # Refresh relationship types.
            self._refresh_relations()
            relationship_type = self._relationship_types[idx]
        return relationship_type

    def get_property(self, idx):
        """
        Returns a property by it's index

        Args:

        idx:
            The index of the property
        """
        try:
            p = self._properties[idx]
        except IndexError:
            # Refresh properties.
            self._refresh_attributes()
            p = self._properties[idx]
        return p

    def add_node(self, node):
        """
        Adds a node to the graph.
        """
        if node.alias is None:
            node.alias = random_string()
        self.nodes[node.alias] = node

    def add_edge(self, edge):
        """
        Adds an edge to the graph.
        """
        if not (self.nodes[edge.src_node.alias] and self.nodes[edge.dest_node.alias]):
            raise AssertionError("Both edge's end must be in the graph")

        self.edges.append(edge)

    def _build_params_header(self, params):
        if params is None:
            return ""
        if not isinstance(params, dict):
            raise TypeError("'params' must be a dict")
        # Header starts with "CYPHER"
        params_header = "CYPHER "
        for key, value in params.items():
            params_header += str(key) + "=" + stringify_param_value(value) + " "
        return params_header

    # Procedures.
    def call_procedure(self, procedure, *args, read_only=False, **kwagrs):
        args = [quote_string(arg) for arg in args]
        q = f"CALL {procedure}({','.join(args)})"

        y = kwagrs.get("y", None)
        if y is not None:
            q += f"YIELD {','.join(y)}"

        return self.query(q, read_only=read_only)

    def labels(self):
        return self.call_procedure(DB_LABELS, read_only=True).result_set

    def relationship_types(self):
        return self.call_procedure(DB_RAELATIONSHIPTYPES, read_only=True).result_set

    def property_keys(self):
        return self.call_procedure(DB_PROPERTYKEYS, read_only=True).result_set


class AsyncGraph(Graph, AsyncGraphCommands):
    """Async version for Graph"""

    async def _refresh_labels(self):
        lbls = await self.labels()

        # Unpack data.
        self._labels = [l[0] for _, l in enumerate(lbls)]

    async def _refresh_attributes(self):
        props = await self.property_keys()

        # Unpack data.
        self._properties = [p[0] for _, p in enumerate(props)]

    async def _refresh_relations(self):
        rels = await self.relationship_types()

        # Unpack data.
        self._relationship_types = [r[0] for _, r in enumerate(rels)]

    async def get_label(self, idx):
        """
        Returns a label by it's index

        Args:

        idx:
            The index of the label
        """
        try:
            label = self._labels[idx]
        except IndexError:
            # Refresh labels.
            await self._refresh_labels()
            label = self._labels[idx]
        return label

    async def get_property(self, idx):
        """
        Returns a property by it's index

        Args:

        idx:
            The index of the property
        """
        try:
            p = self._properties[idx]
        except IndexError:
            # Refresh properties.
            await self._refresh_attributes()
            p = self._properties[idx]
        return p

    async def get_relation(self, idx):
        """
        Returns a relationship type by it's index

        Args:

        idx:
            The index of the relation
        """
        try:
            relationship_type = self._relationship_types[idx]
        except IndexError:
            # Refresh relationship types.
            await self._refresh_relations()
            relationship_type = self._relationship_types[idx]
        return relationship_type

    async def call_procedure(self, procedure, *args, read_only=False, **kwagrs):
        args = [quote_string(arg) for arg in args]
        q = f"CALL {procedure}({','.join(args)})"

        y = kwagrs.get("y", None)
        if y is not None:
            f"YIELD {','.join(y)}"
        return await self.query(q, read_only=read_only)

    async def labels(self):
        return ((await self.call_procedure(DB_LABELS, read_only=True))).result_set

    async def property_keys(self):
        return (await self.call_procedure(DB_PROPERTYKEYS, read_only=True)).result_set

    async def relationship_types(self):
        return (
            await self.call_procedure(DB_RAELATIONSHIPTYPES, read_only=True)
        ).result_set
