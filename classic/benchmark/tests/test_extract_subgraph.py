import pytest

from agbenchmark.utils.dependencies.graphs import extract_subgraph_based_on_category


@pytest.fixture
def curriculum_graph():
    return {
        "edges": [
            {"from": "Calculus", "to": "Advanced Calculus"},
            {"from": "Algebra", "to": "Calculus"},
            {"from": "Biology", "to": "Advanced Biology"},
            {"from": "World History", "to": "Modern History"},
        ],
        "nodes": [
            {"data": {"category": ["math"]}, "id": "Calculus", "label": "Calculus"},
            {
                "data": {"category": ["math"]},
                "id": "Advanced Calculus",
                "label": "Advanced Calculus",
            },
            {"data": {"category": ["math"]}, "id": "Algebra", "label": "Algebra"},
            {"data": {"category": ["science"]}, "id": "Biology", "label": "Biology"},
            {
                "data": {"category": ["science"]},
                "id": "Advanced Biology",
                "label": "Advanced Biology",
            },
            {
                "data": {"category": ["history"]},
                "id": "World History",
                "label": "World History",
            },
            {
                "data": {"category": ["history"]},
                "id": "Modern History",
                "label": "Modern History",
            },
        ],
    }


graph_example = {
    "nodes": [
        {"id": "A", "data": {"category": []}},
        {"id": "B", "data": {"category": []}},
        {"id": "C", "data": {"category": ["math"]}},
    ],
    "edges": [{"from": "B", "to": "C"}, {"from": "A", "to": "C"}],
}


def test_dfs_category_math(curriculum_graph):
    result_graph = extract_subgraph_based_on_category(curriculum_graph, "math")

    # Expected nodes: Algebra, Calculus, Advanced Calculus
    # Expected edges: Algebra->Calculus, Calculus->Advanced Calculus

    expected_nodes = ["Algebra", "Calculus", "Advanced Calculus"]
    expected_edges = [
        {"from": "Algebra", "to": "Calculus"},
        {"from": "Calculus", "to": "Advanced Calculus"},
    ]

    assert set(node["id"] for node in result_graph["nodes"]) == set(expected_nodes)
    assert set((edge["from"], edge["to"]) for edge in result_graph["edges"]) == set(
        (edge["from"], edge["to"]) for edge in expected_edges
    )


def test_extract_subgraph_math_category():
    subgraph = extract_subgraph_based_on_category(graph_example, "math")
    assert set(
        (node["id"], tuple(node["data"]["category"])) for node in subgraph["nodes"]
    ) == set(
        (node["id"], tuple(node["data"]["category"])) for node in graph_example["nodes"]
    )
    assert set((edge["from"], edge["to"]) for edge in subgraph["edges"]) == set(
        (edge["from"], edge["to"]) for edge in graph_example["edges"]
    )


def test_extract_subgraph_non_existent_category():
    result_graph = extract_subgraph_based_on_category(graph_example, "toto")

    # Asserting that the result graph has no nodes and no edges
    assert len(result_graph["nodes"]) == 0
    assert len(result_graph["edges"]) == 0
