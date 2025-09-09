#!/usr/bin/env python
"""Test credential detection in a graph."""

import asyncio
import json

from backend.data import graph as graph_db


async def main():
    # Test with a known graph that requires credentials
    # Replace with an actual graph ID that needs credentials
    test_graph_id = "e02b8123-2566-4240-9e8c-a33cbd27d882"

    try:
        # Get the graph
        graph = await graph_db.get_graph(
            graph_id=test_graph_id,
            version=1,
            user_id=None,
            include_subgraphs=True,
        )

        if graph:
            print(f"Graph: {graph.name}")
            print(f"Graph ID: {graph.id}")

            # Check credentials schema
            if hasattr(graph, "credentials_input_schema"):
                print("\nCredentials Input Schema:")
                print(json.dumps(graph.credentials_input_schema, indent=2))

                # Parse like run_agent does
                credentials_to_check = {}
                if isinstance(graph.credentials_input_schema, dict):
                    if "properties" in graph.credentials_input_schema:
                        credentials_to_check = graph.credentials_input_schema[
                            "properties"
                        ]
                    else:
                        credentials_to_check = graph.credentials_input_schema

                print(
                    f"\nCredential keys required: {list(credentials_to_check.keys())}"
                )

                for cred_key, cred_schema in credentials_to_check.items():
                    print(f"\n{cred_key}:")
                    print(f"  Schema: {cred_schema}")

                    if isinstance(cred_schema, dict):
                        if "credentials_provider" in cred_schema:
                            providers = cred_schema["credentials_provider"]
                            print(f"  Providers: {providers}")
                            if isinstance(providers, list) and len(providers) > 0:
                                provider_name = str(providers[0])
                                if "ProviderName." in provider_name:
                                    provider_name = (
                                        provider_name.split("'")[1]
                                        if "'" in provider_name
                                        else provider_name.split(".")[-1].lower()
                                    )
                                print(f"  Extracted provider: {provider_name}")
            else:
                print("No credentials_input_schema found")

            # Check nodes for credential requirements
            if hasattr(graph, "nodes"):
                print(f"\n{len(graph.nodes)} nodes in graph")
                for node in graph.nodes:
                    if hasattr(node, "input_schema") and node.input_schema:
                        # Check if any inputs are credentials
                        for key, schema in node.input_schema.get(
                            "properties", {}
                        ).items():
                            if "credentials" in str(schema).lower():
                                print(f"  Node {node.id} needs credential: {key}")
                                print(f"    Schema: {schema}")
        else:
            print(f"Graph {test_graph_id} not found")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
