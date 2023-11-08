from agbenchmark.utils.dependencies.graphs import get_roots


def test_get_roots():
    graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
        ],
    }

    result = get_roots(graph)
    assert set(result) == {
        "A",
        "D",
    }, f"Expected roots to be 'A' and 'D', but got {result}"


def test_no_roots():
    fully_connected_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "A"},
        ],
    }

    result = get_roots(fully_connected_graph)
    assert not result, "Expected no roots, but found some"


# def test_no_rcoots():
#     fully_connected_graph = {
#         "nodes": [
#             {"id": "A", "data": {"category": []}},
#             {"id": "B", "data": {"category": []}},
#             {"id": "C", "data": {"category": []}},
#         ],
#         "edges": [
#             {"from": "A", "to": "B"},
#             {"from": "D", "to": "C"},
#         ],
#     }
#
#     result = get_roots(fully_connected_graph)
#     assert set(result) == {"A"}, f"Expected roots to be 'A', but got {result}"
