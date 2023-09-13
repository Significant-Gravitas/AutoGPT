import json
import os
from typing import Optional

import requests

from agbenchmark.__main__ import BENCHMARK_START_TIME
from agbenchmark.agent_interface import HELICONE_GRAPHQL_LOGS


def get_data_from_helicone(challenge: str) -> Optional[float]:
    # Define the endpoint of your GraphQL server
    url = "https://www.helicone.ai/api/graphql"

    # Set the headers, usually you'd need to set the content type and possibly an authorization token
    headers = {"authorization": f"Bearer {os.environ.get('HELICONE_API_KEY')}"}

    # Define the query, variables, and operation name
    query = """
query ExampleQuery($properties: [PropertyFilter!]){
  aggregatedHeliconeRequest(properties: $properties) {
    costUSD
  }
}
"""

    variables = {
        "properties": [
            {
                "value": {"equals": os.environ.get("AGENT_NAME")},
                "name": "agent",
            },
            {
                "value": {"equals": BENCHMARK_START_TIME},
                "name": "benchmark_start_time",
            },
            {"value": {"equals": challenge}, "name": "challenge"},
        ]
    }
    if HELICONE_GRAPHQL_LOGS:
        print(query)
        print(json.dumps(variables, indent=4))

    operation_name = "ExampleQuery"

    data = {}
    response = None

    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "query": query,
                "variables": variables,
                "operationName": operation_name,
            },
        )

        data = response.json()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None  # Re-raise the exception to stop execution
    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response.text if response else 'No response'}")
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return None

    try:
        if data is None or data.get("data") is None:
            print("Invalid response received from server: no data")
            return None
        return (
            data.get("data", {})
            .get("aggregatedHeliconeRequest", {})
            .get("costUSD", None)
        )
    except Exception as err:
        print(f"Error occurred while parsing response: {err}")
        return None
