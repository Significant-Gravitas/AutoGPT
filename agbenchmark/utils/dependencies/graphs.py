import math
from pathlib import Path
import json
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network

from agbenchmark.generate_test import DATA_CATEGORY


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
        src = np.array(pos[u])
        dst = np.array(pos[v])

        same_level = abs(src[1] - dst[1]) < 0.01

        if same_level:
            control = [(src[0] + dst[0]) / 2, src[1] + dist]
            curve = bezier_curve(src, control, dst)
            arrow = patches.FancyArrowPatch(
                posA=curve[0],  # type: ignore
                posB=curve[-1],  # type: ignore
                connectionstyle=f"arc3,rad=0.2",
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
                xy=dst,
                xytext=src,
                arrowprops=dict(
                    arrowstyle="-|>", color="gray", lw=1, shrinkA=10, shrinkB=10
                ),
            )


def tree_layout(graph: nx.DiGraph, root_node: Any) -> Dict[Any, Tuple[float, float]]:
    """Compute positions as a tree layout centered on the root with alternating vertical shifts."""
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
    num_nodes = len(dag.nodes())
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
    dag: nx.DiGraph, labels: Dict[Any, Dict[str, Any]], show: bool = False
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
            print(
                f"Skipping edge {source_id_str} -> {target_id_str} due to missing nodes."
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

    # Serialize the graph to JSON
    graph_data = {"nodes": nt.nodes, "edges": nt.edges}

    json_graph = json.dumps(graph_data)

    # Optionally, save to a file
    with open(Path("agbenchmark/challenges/skill-tree/graph.json").resolve(), "w") as f:
        f.write(json_graph)

    relative_path = "agbenchmark/challenges/skill-tree/index.html"
    file_path = str(Path(relative_path).resolve())

    if show:
        nt.show(file_path, notebook=False)
    nt.write_html(file_path)

    # Example usage
    table_data = [
        ["Task: ", "Click on a skill to to see the task"],
    ]

    iframe_path = "index.html"
    combined_file_path = "agbenchmark/challenges/skill-tree/combined_view.html"

    create_combined_html(combined_file_path, iframe_path, table_data)
    # JavaScript code snippet to be inserted
    iframe_js_code = """
    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            var clickedNodeId = params.nodes[0];
            var clickedNode = nodes.get(clickedNodeId);
            var clickedNodeLabel = clickedNode.task;
            window.parent.updateLabel(clickedNodeLabel);
        }
    });
    """

    # Path to the iframe HTML file
    iframe_path = "agbenchmark/challenges/skill-tree/index.html"

    # Insert the JS code snippet into the iframe HTML file
    insert_js_into_iframe(iframe_path, iframe_js_code)


def create_combined_html(
    file_path: str, iframe_path: str, table_data: List[List[Any]]
) -> None:
    table_html = "<table>"
    for row in table_data:
        table_html += "<tr>"
        for cell in row:
            table_html += f"<td>{cell}</td>"
        table_html += "</tr>"
    table_html += "</table>"
    table_html = table_html.replace(
        "<td>Click on a skill to to see the task</td>",
        '<td id="labelCell">Click on a skill to to see the task</td>',
        1,
    )

    # JavaScript function to update the table
    js_function = """
    <script type="text/javascript">
        function updateLabel(label) {
            document.getElementById('labelCell').innerText = label;
        }
    </script>
    """

    iframe_html = f'<iframe src="{iframe_path}" width="100%" height="800px"></iframe>'

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Graph with Table</title>
    </head>
    <body>
        {js_function}
        {table_html}
        {iframe_html}
    </body>
    </html>
    """

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(full_html)


def insert_js_into_iframe(iframe_path: str, js_code: str) -> None:
    with open(iframe_path, "r", encoding="utf-8") as file:
        content = file.readlines()

    # Locate the line number where "drawGraph();" is called
    line_number = -1
    for index, line in enumerate(content):
        if "drawGraph();" in line:
            line_number = index
            break

    # Insert the JS code snippet just after "drawGraph();"
    if line_number != -1:
        content.insert(line_number + 1, js_code)

    with open(iframe_path, "w", encoding="utf-8") as file:
        file.writelines(content)
