import json
import os
from typing import Optional

import requests

from agbenchmark.start_benchmark import BENCHMARK_START_TIME


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
    print(query)

    variables = {
        "filters": [
            {
                "property": {
                    "value": {"equals": os.environ.get("AGENT_NAME")},
                    "name": "agent",
                }
            },
            {
                "property": {
                    "value": {"equals": BENCHMARK_START_TIME},
                    "name": "benchmark_start_time",
                }
            },
            {"property": {"value": {"equals": challenge}, "name": "challenge"}},
        ]
    }
    print(json.dumps(variables, indent=4))

    operation_name = "ExampleQuery"

    data = None
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
        response.raise_for_status()  # Raises a HTTPError if the response was an unsuccessful status code

        data = response.json()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        raise  # Re-raise the exception to stop execution
    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response.text if response else 'No response'}")
        raise
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise

    if data is None or data.get("data") is None:
        raise ValueError("Invalid response received from server: no data")

    return (
        data.get("data", {}).get("aggregatedHeliconeRequest", {}).get("costUSD", None)
    )
