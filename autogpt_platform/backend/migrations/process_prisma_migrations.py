import argparse
import re


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
    if_not_exists_statements = [table_block] + indexes + constraints
    indented_statements = "\n\n".join(
        [indent_block(stmt.strip(), 8) for stmt in if_not_exists_statements]
    )

    table_name_match = re.search(r'CREATE TABLE "([^"]+)"', table_block)
    table_name = table_name_match.group(1) if table_name_match else None

    wrapped_block = f"""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = '{table_name}') THEN
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
    IF NOT EXISTS (SELECT 1 FROM {block_type} WHERE {condition}) THEN
{indent_block(block.strip(), 8)}
    END IF;
END $$;"""
    return wrapped_block.strip()


def process_sql_file(input_file: str, output_file: str):
    """
    Process an SQL file to:
    - Leave ENUMs at the top.
    - For each table, find the indexes and constraints related to that table and add them below the table definition.
    - Add IF NOT EXISTS checks to relevant SQL statements, with proper indentation.
    - Wrap all indexes and constraints related to a table in a single IF NOT EXISTS block.
    """
    with open(input_file, "r") as infile:
        sql_content = infile.read()

    # Split the SQL file by double newlines into blocks
    blocks = sql_content.split("\n\n")

    # Separate blocks into enums, tables, indexes, and constraints
    enums = []
    tables = {}
    standalone_indexes = []
    standalone_constraints = []
    table_related_indexes = {}
    table_related_constraints = {}

    # Classify each block
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if "CREATE TYPE" in block:
            enum_name_match = re.search(r'CREATE TYPE "([^"]+)"', block)
            enum_name = enum_name_match.group(1) if enum_name_match else None
            if not enum_name:
                continue
            enums.append(
                wrap_standalone_block(block, "pg_type", f"typname = '{enum_name}'")
            )
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
                standalone_indexes.append(
                    wrap_standalone_block(
                        block, "pg_indexes", f"indexname = '{index_name}'"
                    )
                )
        elif "ADD CONSTRAINT" in block and "FOREIGN KEY" in block:
            table_match = re.search(r'ALTER TABLE "([^"]+)"', block)
            if table_match:
                table_name = table_match.group(1)
                if table_name not in table_related_constraints:
                    table_related_constraints[table_name] = []
                table_related_constraints[table_name].append(block)
            else:
                constraint_name_match = re.search(r'ADD CONSTRAINT "([^"]+)"', block)
                constraint_name = (
                    constraint_name_match.group(1) if constraint_name_match else None
                )
                if not constraint_name:
                    continue
                standalone_constraints.append(
                    wrap_standalone_block(
                        block, "pg_constraint", f"conname = '{constraint_name}'"
                    )
                )

    # Construct the final output SQL
    final_sql = "BEGIN;\n\n"

    # Add all enums to the top
    final_sql += "\n\n".join(enums) + "\n\n"

    # Add each table with its related indexes and constraints
    for table_name, table_block in tables.items():
        final_sql += "-" * 100 + "\n"
        final_sql += f"-- Table: {table_name}\n"
        final_sql += "-" * 100 + "\n\n"
        related_indexes = table_related_indexes.get(table_name, [])
        related_constraints = table_related_constraints.get(table_name, [])
        final_sql += (
            wrap_table_with_indexes_and_constraints(
                table_block, related_indexes, related_constraints
            )
            + "\n\n"
        )

    # Add standalone indexes and constraints that were not tied to a table
    final_sql += "\n\n".join(standalone_indexes) + "\n\n"
    final_sql += "\n\n".join(standalone_constraints) + "\n\n"

    # Add the closing COMMIT statement
    final_sql += "COMMIT;"

    # Write the output SQL to a new file
    with open(output_file, "w") as outfile:
        outfile.write(final_sql)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Prisma generated migration files.")
    parser.add_argument("input_file", help="The input SQL migration file name.")
    parser.add_argument("output_file", help="The desired output file name.")
    args = parser.parse_args()

    process_sql_file(args.input_file, args.output_file)
    print(f"Processed SQL written to {args.output_file}")
