import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network

from agbenchmark.generate_test import DATA_CATEGORY
from agbenchmark.utils.utils import write_pretty_json

logger = logging.getLogger(__name__)


def bezier_curve(
    src: np.ndarray, ctrl: List[float], dst: np.ndarray
) -> List[np.ndarray]:
    """
    Generate Bézier curve points.

    Args:
    - src (np.ndarray): The source point.
    - ctrl (List[float]): The control point.
    - dst (np.ndarray): The destination point.

    Returns:
    - List[np.ndarray]: The Bézier curve points.
    """
    curve = []
    for t in np.linspace(0, 1, num=100):
        curve_point = (
            np.outer((1 - t) ** 2, src)
            + 2 * np.outer((1 - t) * t, ctrl)
            + np.outer(t**2, dst)
        )
        curve.append(curve_point[0])
    return curve


def curved_edges(
    G: nx.Graph, pos: Dict[Any, Tuple[float, float]], dist: float = 0.2
) -> None:
    """
    Draw curved edges for nodes on the same level.

    Args:
    - G (Any): The graph object.
    - pos (Dict[Any, Tuple[float, float]]): Dictionary with node positions.
    - dist (float, optional): Distance for curvature. Defaults to 0.2.

    Returns:
    - None
    """
    ax = plt.gca()
    for u, v, data in G.edges(data=True):
        _src = pos[u]
        _dst = pos[v]
        src = np.array(_src)
        dst = np.array(_dst)

        same_level = abs(src[1] - dst[1]) < 0.01

        if same_level:
            control = [(src[0] + dst[0]) / 2, src[1] + dist]
            curve = bezier_curve(src, control, dst)
            arrow = patches.FancyArrowPatch(
                posA=curve[0],  # type: ignore
                posB=curve[-1],  # type: ignore
                connectionstyle="arc3,rad=0.2",
                color="gray",
                arrowstyle="-|>",
                mutation_scale=15.0,
                lw=1,
                shrinkA=10,
                shrinkB=10,
            )
            ax.add_patch(arrow)
        else:
            ax.annotate(
                "",
                xy=_dst,
                xytext=_src,
                arrowprops=dict(
                    arrowstyle="-|>", color="gray", lw=1, shrinkA=10, shrinkB=10
                ),
            )


def tree_layout(graph: nx.DiGraph, root_node: Any) -> Dict[Any, Tuple[float, float]]:
    """Compute positions as a tree layout centered on the root
    with alternating vertical shifts."""
    bfs_tree = nx.bfs_tree(graph, source=root_node)
    levels = {
        node: depth
        for node, depth in nx.single_source_shortest_path_length(
            bfs_tree, root_node
        ).items()
    }

    pos = {}
    max_depth = max(levels.values())
    level_positions = {i: 0 for i in range(max_depth + 1)}  # type: ignore

    # Count the number of nodes per level to compute the width
    level_count: Any = {}
    for node, level in levels.items():
        level_count[level] = level_count.get(level, 0) + 1

    vertical_offset = (
        0.07  # The amount of vertical shift per node within the same level
    )

    # Assign positions
    for node, level in sorted(levels.items(), key=lambda x: x[1]):
        total_nodes_in_level = level_count[level]
        horizontal_spacing = 1.0 / (total_nodes_in_level + 1)
        pos_x = (
            0.5
            - (total_nodes_in_level - 1) * horizontal_spacing / 2
            + level_positions[level] * horizontal_spacing
        )

        # Alternately shift nodes up and down within the same level
        pos_y = (
            -level
            + (level_positions[level] % 2) * vertical_offset
            - ((level_positions[level] + 1) % 2) * vertical_offset
        )
        pos[node] = (pos_x, pos_y)

        level_positions[level] += 1

    return pos


def graph_spring_layout(
    dag: nx.DiGraph, labels: Dict[Any, str], tree: bool = True
) -> None:
    num_nodes = len(list(dag.nodes()))
    # Setting up the figure and axis
    fig, ax = plt.subplots()
    ax.axis("off")  # Turn off the axis

    base = 3.0

    if num_nodes > 10:
        base /= 1 + math.log(num_nodes)
        font_size = base * 10

    font_size = max(10, base * 10)
    node_size = max(300, base * 1000)

    if tree:
        root_node = [node for node, degree in dag.in_degree() if degree == 0][0]
        pos = tree_layout(dag, root_node)
    else:
        # Adjust k for the spring layout based on node count
        k_value = 3 / math.sqrt(num_nodes)

        pos = nx.spring_layout(dag, k=k_value, iterations=50)

    # Draw nodes and labels
    nx.draw_networkx_nodes(dag, pos, node_color="skyblue", node_size=int(node_size))
    nx.draw_networkx_labels(dag, pos, labels=labels, font_size=int(font_size))

    # Draw curved edges
    curved_edges(dag, pos)  # type: ignore

    plt.tight_layout()
    plt.show()


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def get_category_colors(categories: Dict[Any, str]) -> Dict[str, str]:
    unique_categories = set(categories.values())
    colormap = plt.cm.get_cmap("tab10", len(unique_categories))  # type: ignore
    return {
        category: rgb_to_hex(colormap(i)[:3])
        for i, category in enumerate(unique_categories)
    }


def graph_interactive_network(
    dag: nx.DiGraph,
    labels: Dict[Any, Dict[str, Any]],
    html_graph_path: str = "",
) -> None:
    nt = Network(notebook=True, width="100%", height="800px", directed=True)

    category_colors = get_category_colors(DATA_CATEGORY)

    # Add nodes and edges to the pyvis network
    for node, json_data in labels.items():
        label = json_data.get("name", "")
        # remove the first 4 letters of label
        label_without_test = label[4:]
        node_id_str = node.nodeid

        # Get the category for this label
        category = DATA_CATEGORY.get(
            label, "unknown"
        )  # Default to 'unknown' if label not found

        # Get the color for this category
        color = category_colors.get(category, "grey")

        nt.add_node(
            node_id_str,
            label=label_without_test,
            color=color,
            data=json_data,
        )

    # Add edges to the pyvis network
    for edge in dag.edges():
        source_id_str = edge[0].nodeid
        target_id_str = edge[1].nodeid
        edge_id_str = (
            f"{source_id_str}_to_{target_id_str}"  # Construct a unique edge id
        )
        if not (source_id_str in nt.get_nodes() and target_id_str in nt.get_nodes()):
            logger.warning(
                f"Skipping edge {source_id_str} -> {target_id_str} due to missing nodes"
            )
            continue
        nt.add_edge(source_id_str, target_id_str, id=edge_id_str)

    # Configure physics for hierarchical layout
    hierarchical_options = {
        "enabled": True,
        "levelSeparation": 200,  # Increased vertical spacing between levels
        "nodeSpacing": 250,  # Increased spacing between nodes on the same level
        "treeSpacing": 250,  # Increased spacing between different trees (for forest)
        "blockShifting": True,
        "edgeMinimization": True,
        "parentCentralization": True,
        "direction": "UD",
        "sortMethod": "directed",
    }

    physics_options = {
        "stabilization": {
            "enabled": True,
            "iterations": 1000,  # Default is often around 100
        },
        "hierarchicalRepulsion": {
            "centralGravity": 0.0,
            "springLength": 200,  # Increased edge length
            "springConstant": 0.01,
            "nodeDistance": 250,  # Increased minimum distance between nodes
            "damping": 0.09,
        },
        "solver": "hierarchicalRepulsion",
        "timestep": 0.5,
    }

    nt.options = {
        "nodes": {
            "font": {
                "size": 20,  # Increased font size for labels
                "color": "black",  # Set a readable font color
            },
            "shapeProperties": {"useBorderWithImage": True},
        },
        "edges": {
            "length": 250,  # Increased edge length
        },
        "physics": physics_options,
        "layout": {"hierarchical": hierarchical_options},
    }

    # Serialize the graph to JSON and save in appropriate locations
    graph_data = {"nodes": nt.nodes, "edges": nt.edges}
    logger.debug(f"Generated graph data:\n{json.dumps(graph_data, indent=4)}")

    # FIXME: use more reliable method to find the right location for these files.
    #   This will fail in all cases except if run from the root of our repo.
    home_path = Path.cwd()
    write_pretty_json(graph_data, home_path / "frontend" / "public" / "graph.json")

    flutter_app_path = home_path.parent / "frontend" / "assets"

    # Optionally, save to a file
    # Sync with the flutter UI
    # this literally only works in the AutoGPT repo, but this part of the code
    # is not reached if BUILD_SKILL_TREE is false
    write_pretty_json(graph_data, flutter_app_path / "tree_structure.json")
    validate_skill_tree(graph_data, "")

    # Extract node IDs with category "coding"

    coding_tree = extract_subgraph_based_on_category(graph_data.copy(), "coding")
    validate_skill_tree(coding_tree, "coding")
    write_pretty_json(
        coding_tree,
        flutter_app_path / "coding_tree_structure.json",
    )

    data_tree = extract_subgraph_based_on_category(graph_data.copy(), "data")
    # validate_skill_tree(data_tree, "data")
    write_pretty_json(
        data_tree,
        flutter_app_path / "data_tree_structure.json",
    )

    general_tree = extract_subgraph_based_on_category(graph_data.copy(), "general")
    validate_skill_tree(general_tree, "general")
    write_pretty_json(
        general_tree,
        flutter_app_path / "general_tree_structure.json",
    )

    scrape_synthesize_tree = extract_subgraph_based_on_category(
        graph_data.copy(), "scrape_synthesize"
    )
    validate_skill_tree(scrape_synthesize_tree, "scrape_synthesize")
    write_pretty_json(
        scrape_synthesize_tree,
        flutter_app_path / "scrape_synthesize_tree_structure.json",
    )

    if html_graph_path:
        file_path = str(Path(html_graph_path).resolve())

        nt.write_html(file_path)


def extract_subgraph_based_on_category(graph, category):
    """
    Extracts a subgraph that includes all nodes and edges required to reach all nodes
    with a specified category.

    :param graph: The original graph.
    :param category: The target category.
    :return: Subgraph with nodes and edges required to reach the nodes
        with the given category.
    """

    subgraph = {"nodes": [], "edges": []}
    visited = set()

    def reverse_dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)

        node_data = next(node for node in graph["nodes"] if node["id"] == node_id)

        # Add the node to the subgraph if it's not already present.
        if node_data not in subgraph["nodes"]:
            subgraph["nodes"].append(node_data)

        for edge in graph["edges"]:
            if edge["to"] == node_id:
                if edge not in subgraph["edges"]:
                    subgraph["edges"].append(edge)
                reverse_dfs(edge["from"])

    # Identify nodes with the target category and initiate reverse DFS from them.
    nodes_with_target_category = [
        node["id"] for node in graph["nodes"] if category in node["data"]["category"]
    ]

    for node_id in nodes_with_target_category:
        reverse_dfs(node_id)

    return subgraph


def is_circular(graph):
    def dfs(node, visited, stack, parent_map):
        visited.add(node)
        stack.add(node)
        for edge in graph["edges"]:
            if edge["from"] == node:
                if edge["to"] in stack:
                    # Detected a cycle
                    cycle_path = []
                    current = node
                    while current != edge["to"]:
                        cycle_path.append(current)
                        current = parent_map.get(current)
                    cycle_path.append(edge["to"])
                    cycle_path.append(node)
                    return cycle_path[::-1]
                elif edge["to"] not in visited:
                    parent_map[edge["to"]] = node
                    cycle_path = dfs(edge["to"], visited, stack, parent_map)
                    if cycle_path:
                        return cycle_path
        stack.remove(node)
        return None

    visited = set()
    stack = set()
    parent_map = {}
    for node in graph["nodes"]:
        node_id = node["id"]
        if node_id not in visited:
            cycle_path = dfs(node_id, visited, stack, parent_map)
            if cycle_path:
                return cycle_path
    return None


def get_roots(graph):
    """
    Return the roots of a graph. Roots are nodes with no incoming edges.
    """
    # Create a set of all node IDs
    all_nodes = {node["id"] for node in graph["nodes"]}

    # Create a set of nodes with incoming edges
    nodes_with_incoming_edges = {edge["to"] for edge in graph["edges"]}

    # Roots are nodes that have no incoming edges
    roots = all_nodes - nodes_with_incoming_edges

    return list(roots)


def validate_skill_tree(graph, skill_tree_name):
    """
    Validate if a given graph represents a valid skill tree
    and raise appropriate exceptions if not.

    :param graph: A dictionary representing the graph with 'nodes' and 'edges'.
    :raises: ValueError with a description of the invalidity.
    """
    # Check for circularity
    cycle_path = is_circular(graph)
    if cycle_path:
        cycle_str = " -> ".join(cycle_path)
        raise ValueError(
            f"{skill_tree_name} skill tree is circular! "
            f"Detected circular path: {cycle_str}."
        )

    # Check for multiple roots
    roots = get_roots(graph)
    if len(roots) > 1:
        raise ValueError(f"{skill_tree_name} skill tree has multiple roots: {roots}.")
    elif not roots:
        raise ValueError(f"{skill_tree_name} skill tree has no roots.")
