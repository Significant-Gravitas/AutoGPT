import argparse
import re
from collections import defaultdict


def indent_block(block: str, indent_level: int = 4) -> str:
    """
    Indent each line of a block by a specified number of spaces.
    """
    indent = " " * indent_level
    indented_lines = [
        indent + line if line.strip() else line for line in block.splitlines()
    ]
    return "\n".join(indented_lines)


def wrap_table_with_indexes_and_constraints(
    table_block: str, indexes: list, constraints: list
) -> str:
    """
    Wrap a table definition with its related indexes and constraints in a single DO block.
    """
    # Add closing bracket if missing
    if not table_block.strip().endswith(");"):
        table_block = table_block.rstrip() + "\n);"

    if_not_exists_statements = [table_block] + indexes + constraints
    indented_statements = "\n\n".join(
        [indent_block(stmt.strip(), 8) for stmt in if_not_exists_statements]
    )

    table_name_match = re.search(r'CREATE TABLE "([^"]+)"', table_block)
    table_name = table_name_match.group(1) if table_name_match else None

    wrapped_block = f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = '{table_name}' AND schemaname = CURRENT_SCHEMA()) THEN
{indented_statements}
    END IF;
END $$;"""
    return wrapped_block.strip()


def wrap_standalone_block(block: str, block_type: str, condition: str) -> str:
    """
    Wrap standalone blocks like indexes or constraints in their own IF NOT EXISTS block.
    """
    wrapped_block = f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM {block_type} WHERE {condition} AND schemaname = CURRENT_SCHEMA()) THEN
{indent_block(block.strip(), 8)}
    END IF;
END $$;"""
    return wrapped_block.strip()


def topological_sort(nodes, edges):
    """
    Perform a deterministic topological sort on the given nodes and edges.
    nodes: a set of node names.
    edges: a dict mapping from node to a set of nodes it depends on.
    Returns a list of nodes in topological order.
    Raises an exception if there is a cycle.
    """
    in_degree = defaultdict(int)
    graph = defaultdict(set)
    for node in nodes:
        in_degree[node] = 0
    for node in edges:
        for dep in edges[node]:
            graph[dep].add(node)
            in_degree[node] += 1

    # Use a list to store nodes with zero in-degree, and sort it to ensure determinism
    zero_in_degree_nodes = sorted([node for node in nodes if in_degree[node] == 0])

    sorted_list = []

    while zero_in_degree_nodes:
        node = zero_in_degree_nodes.pop(0)
        sorted_list.append(node)
        for m in sorted(graph[node]):
            in_degree[m] -= 1
            if in_degree[m] == 0:
                # Insert node and keep the list sorted for determinism
                zero_in_degree_nodes.append(m)
                zero_in_degree_nodes.sort()

    if len(sorted_list) != len(nodes):
        raise Exception("Cycle detected in dependency graph")
    return sorted_list


def detect_cycles_and_remove_edges(nodes, edges, edge_to_constraint):
    """
    Detect cycles in the dependency graph and remove edges to break cycles.
    Returns a list of foreign key constraints that need to be deferred.
    """
    edges_copy = {node: set(deps) for node, deps in edges.items()}  # Copy edges
    deferred_constraints = []
    removed_edges = set()

    while True:
        try:
            sorted_nodes = topological_sort(nodes, edges_copy)
            break  # If topological sort succeeds, exit the loop
        except Exception:
            # If a cycle is detected, find cycles and remove one edge from each
            cycles = find_cycles(edges_copy)
            if not cycles:
                raise Exception("Cycle detected but no cycles found in graph.")
            for cycle in cycles:
                if len(cycle) >= 2:
                    # Remove the edge from the last node to the first node in the cycle
                    u = cycle[-1]
                    v = cycle[0]
                    edge = (u, v)
                    if edge in edge_to_constraint:
                        deferred_constraints.append(edge_to_constraint[edge])
                    if v in edges_copy[u]:
                        edges_copy[u].remove(v)
                        removed_edges.add(edge)
                else:
                    # Cycle of length 1 (self-loop), remove it
                    node = cycle[0]
                    edges_copy[node].remove(node)
                    edge = (node, node)
                    if edge in edge_to_constraint:
                        deferred_constraints.append(edge_to_constraint[edge])
                    removed_edges.add(edge)
    return sorted_nodes, deferred_constraints


def find_cycles(edges):
    """
    Find cycles in the graph using Tarjan's algorithm.
    Returns a list of cycles, where each cycle is a list of nodes.
    """
    index = 0
    index_stack = []
    lowlinks = {}
    index_dict = {}
    on_stack = set()
    cycles = []

    def strongconnect(node):
        nonlocal index
        index_dict[node] = index
        lowlinks[node] = index
        index += 1
        index_stack.append(node)
        on_stack.add(node)

        for neighbor in edges.get(node, []):
            if neighbor not in index_dict:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], index_dict[neighbor])

        # If node is a root node, pop the stack and generate a SCC (Strongly Connected Component)
        if lowlinks[node] == index_dict[node]:
            scc = []
            while True:
                w = index_stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == node:
                    break
            if len(scc) > 1 or (len(scc) == 1 and node in edges.get(node, [])):
                # It's a cycle
                cycles.append(scc)

    for node in edges:
        if node not in index_dict:
            strongconnect(node)

    return cycles


def process_sql_file(input_file: str, output_file: str):
    """
    Process an SQL file to:
    - Leave ENUMs at the top.
    - For each table, find the indexes and constraints related to that table and add them below the table definition.
    - Add IF NOT EXISTS checks to relevant SQL statements, with proper indentation.
    - Wrap all indexes and constraints related to a table in a single IF NOT EXISTS block.
    - Ensure that tables are created before they are referenced in foreign key constraints.
    - Extract cyclic dependency foreign keys and add them at the end of the file.
    """
    with open(input_file, "r") as infile:
        sql_content = infile.read()

    # Split the SQL file by semicolons into blocks
    blocks = sql_content.split(";")

    # Separate blocks into enums, tables, indexes, and constraints
    enums = []
    tables = {}
    standalone_indexes = []
    standalone_constraints = []
    table_related_indexes = {}
    table_related_constraints = {}
    table_dependencies = defaultdict(set)
    edge_to_constraint = {}

    # Classify each block
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block += ";"
        if "CREATE TYPE" in block:
            enum_name_match = re.search(r'CREATE TYPE "([^"]+)"', block)
            enum_name = enum_name_match.group(1) if enum_name_match else None
            if not enum_name:
                continue
            # enums.append(
            #     wrap_standalone_block(block, "pg_type", f"typname = '{enum_name}'")
            # )
            enums.append(block)
        elif "CREATE TABLE" in block:
            table_name_match = re.search(r'CREATE TABLE "([^"]+)"', block)
            table_name = table_name_match.group(1) if table_name_match else None
            if not table_name:
                continue
            tables[table_name] = block
        elif "CREATE INDEX" in block or "CREATE UNIQUE INDEX" in block:
            table_match = re.search(r'ON "([^"]+)"', block)
            if table_match:
                table_name = table_match.group(1)
                if table_name not in table_related_indexes:
                    table_related_indexes[table_name] = []
                table_related_indexes[table_name].append(block)
            else:
                index_name_match = re.search(r'CREATE (UNIQUE )?INDEX "([^"]+)"', block)
                index_name = index_name_match.group(2) if index_name_match else None
                if not index_name:
                    continue
                # standalone_indexes.append(
                #     wrap_standalone_block(
                #         block, "pg_indexes", f"indexname = '{index_name}'"
                #     )
                # )
                standalone_indexes.append(block)
        elif "ADD CONSTRAINT" in block and "FOREIGN KEY" in block:
            table_match = re.search(r'ALTER TABLE "([^"]+)"', block)
            if table_match:
                source_table = table_match.group(1)
                if source_table not in table_related_constraints:
                    table_related_constraints[source_table] = []
                table_related_constraints[source_table].append(block)
                # Extract the referenced table
                ref_table_match = re.search(r'REFERENCES "([^"]+)"', block)
                if ref_table_match:
                    referenced_table = ref_table_match.group(1)
                    # Build dependency from source_table to referenced_table if the referenced table is in 'tables'
                    if referenced_table in tables:
                        table_dependencies[source_table].add(referenced_table)
                        edge_to_constraint[(source_table, referenced_table)] = block
            else:
                constraint_name_match = re.search(r'ADD CONSTRAINT "([^"]+)"', block)
                constraint_name = (
                    constraint_name_match.group(1) if constraint_name_match else None
                )
                if not constraint_name:
                    continue
                # standalone_constraints.append(
                #     wrap_standalone_block(
                #         block, "pg_constraint", f"conname = '{constraint_name}'"
                #     )
                # )
                standalone_constraints.append(block)
        else:
            print(f"Unhandled block: {block}")

    all_table_names = set(tables.keys())

    # Detect cycles and remove edges causing cycles
    try:
        sorted_tables, deferred_constraints = detect_cycles_and_remove_edges(
            all_table_names, table_dependencies, edge_to_constraint
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return

    # Update table_related_constraints by removing deferred constraints
    for constraint in deferred_constraints:
        table_match = re.search(r'ALTER TABLE "([^"]+)"', constraint)
        if table_match:
            source_table = table_match.group(1)
            if source_table in table_related_constraints:
                if constraint in table_related_constraints[source_table]:
                    table_related_constraints[source_table].remove(constraint)

    final_sql = "BEGIN;\n\n"

    # Add all enums to the top
    if enums:
        final_sql += "\n\n".join(enums) + "\n\n"

    # Add each table with its related indexes and constraints
    for table_name in sorted_tables:
        table_block = tables[table_name]
        final_sql += "-" * 100 + "\n"
        final_sql += f"-- Table: {table_name}\n"
        final_sql += "-" * 100 + "\n\n"
        related_indexes = table_related_indexes.get(table_name, [])
        related_constraints = table_related_constraints.get(table_name, [])
        # final_sql += (
        #     wrap_table_with_indexes_and_constraints(
        #         table_block, related_indexes, related_constraints
        #     )
        #     + "\n\n"
        # )
        final_sql += table_block + "\n\n"
    # Add standalone indexes and constraints that were not tied to a table
    if standalone_indexes:
        final_sql += "\n\n".join(standalone_indexes) + "\n\n"
    if standalone_constraints:
        final_sql += "\n\n".join(standalone_constraints) + "\n\n"

    # Add deferred foreign key constraints at the end
    if deferred_constraints:
        final_sql += "-" * 100 + "\n\n"
        final_sql += "-- Deferred Foreign Key Constraints (Cyclic Dependencies)\n"
        final_sql += "-" * 100 + "\n\n"
        for constraint in deferred_constraints:
            constraint_name_match = re.search(r'ADD CONSTRAINT "([^"]+)"', constraint)
            constraint_name = (
                constraint_name_match.group(1) if constraint_name_match else None
            )
            if constraint_name:
                # wrapped_constraint = wrap_standalone_block(
                #     constraint, "pg_constraint", f"conname = '{constraint_name}'"
                # )
                wrapped_constraint = constraint 
            else:
                wrapped_constraint = constraint
            final_sql += wrapped_constraint + "\n\n"

    final_sql = final_sql.strip() + "\n\n"
    final_sql += "COMMIT;"

    with open(output_file, "w") as outfile:
        outfile.write(final_sql)

    print(f"Processed SQL written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SQL migration files.")
    parser.add_argument("input_file", help="The input SQL migration file name.")
    parser.add_argument("output_file", help="The desired output file name.")
    args = parser.parse_args()

    process_sql_file(args.input_file, args.output_file)
