import pytest
from agbenchmark.utils.dependencies.graphs import extract_subgraph_based_on_category

# ----- Fixtures -----

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
            {"data": {"category": ["math"]}, "id": "Calculus"},
            {"data": {"category": ["math"]}, "id": "Advanced Calculus"},
            {"data": {"category": ["math"]}, "id": "Algebra"},
            {"data": {"category": ["science"]}, "id": "Biology"},
            {"data": {"category": ["science"]}, "id": "Advanced Biology"},
            {"data": {"category": ["history"]}, "id": "World History"},
            {"data": {"category": ["history"]}, "id": "Modern History"},
        ],
    }

GRAPH_EXAMPLE = {
    "nodes": [
        {"id": "A", "data": {"category": []}},
        {"id": "B", "data": {"category": []}},
        {"id": "C", "data": {"category": ["math"]}},
    ],
    "edges": [{"from": "B", "to": "C"}, {"from": "A", "to": "C"}],
}

# ----- Helper Functions -----

def node_ids_from_graph(graph):
    return set(node["id"] for node in graph["nodes"])

def edge_pairs_from_graph(graph):
    return set((edge["from"], edge["to"]) for edge in graph["edges"])

# ----- Tests -----

def test_dfs_category_math(curriculum_graph):
    result_graph = extract_subgraph_based_on_category(curriculum_graph, "math")

    expected_nodes = {"Algebra", "Calculus", "Advanced Calculus"}
    expected_edges = {("Algebra", "Calculus"), ("Calculus", "Advanced Calculus")}

    assert node_ids_from_graph(result_graph) == expected_nodes
    assert edge_pairs_from_graph(result_graph) == expected_edges

def test_extract_subgraph_math_category():
    subgraph = extract_subgraph_based_on_category(GRAPH_EXAMPLE, "math")
    
    assert node_ids_from_graph(subgraph) == node_ids_from_graph(GRAPH_EXAMPLE)
    assert edge_pairs_from_graph(subgraph) == edge_pairs_from_graph(GRAPH_EXAMPLE)

def test_extract_subgraph_non_existent_category():
    result_graph = extract_subgraph_based_on_category(GRAPH_EXAMPLE, "toto")

    assert node_ids_from_graph(result_graph) == set()
    assert edge_pairs_from_graph(result_graph) == set()
