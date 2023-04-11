from .edge import Edge
from .node import Node


class Path:
    def __init__(self, nodes, edges):
        if not (isinstance(nodes, list) and isinstance(edges, list)):
            raise TypeError("nodes and edges must be list")

        self._nodes = nodes
        self._edges = edges
        self.append_type = Node

    @classmethod
    def new_empty_path(cls):
        return cls([], [])

    def nodes(self):
        return self._nodes

    def edges(self):
        return self._edges

    def get_node(self, index):
        return self._nodes[index]

    def get_relationship(self, index):
        return self._edges[index]

    def first_node(self):
        return self._nodes[0]

    def last_node(self):
        return self._nodes[-1]

    def edge_count(self):
        return len(self._edges)

    def nodes_count(self):
        return len(self._nodes)

    def add_node(self, node):
        if not isinstance(node, self.append_type):
            raise AssertionError("Add Edge before adding Node")
        self._nodes.append(node)
        self.append_type = Edge
        return self

    def add_edge(self, edge):
        if not isinstance(edge, self.append_type):
            raise AssertionError("Add Node before adding Edge")
        self._edges.append(edge)
        self.append_type = Node
        return self

    def __eq__(self, other):
        # Type checking
        if not isinstance(other, Path):
            return False

        return self.nodes() == other.nodes() and self.edges() == other.edges()

    def __str__(self):
        res = "<"
        edge_count = self.edge_count()
        for i in range(0, edge_count):
            node_id = self.get_node(i).id
            res += "(" + str(node_id) + ")"
            edge = self.get_relationship(i)
            res += (
                "-[" + str(int(edge.id)) + "]->"
                if edge.src_node == node_id
                else "<-[" + str(int(edge.id)) + "]-"
            )
        node_id = self.get_node(edge_count).id
        res += "(" + str(node_id) + ")"
        res += ">"
        return res
