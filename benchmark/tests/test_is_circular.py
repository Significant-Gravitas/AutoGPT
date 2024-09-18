from agbenchmark.utils.dependencies.graphs import is_circular


def test_is_circular():
    cyclic_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},  # New node
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
            {"from": "D", "to": "A"},  # This edge creates a cycle
        ],
    }

    result = is_circular(cyclic_graph)
    assert result is not None, "Expected a cycle, but none was detected"
    assert all(
        (
            (result[i], result[i + 1])
            in [(x["from"], x["to"]) for x in cyclic_graph["edges"]]
        )
        for i in range(len(result) - 1)
    ), "The detected cycle path is not part of the graph's edges"


def test_is_not_circular():
    acyclic_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},  # New node
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
            # No back edge from D to any node, so it remains acyclic
        ],
    }

    assert is_circular(acyclic_graph) is None, "Detected a cycle in an acyclic graph"
