import os
from typing import Optional

import requests

from agbenchmark.start_benchmark import BENCHMARK_START_TIME


def get_data_from_helicone(challenge: str) -> Optional[float]:
    # Define the endpoint of your GraphQL server
    url = "https://www.helicone.ai/api/graphql"

    # Set the headers, usually you'd need to set the content type and possibly an authorization token
    headers = {"authorization": "Bearer {os.environ.get('HELICONE_API_KEY')}"}

    # Define the query, variables, and operation name
    query = """
query ExampleQuery($properties: [PropertyFilter!]){
  aggregatedHeliconeRequest(properties: $properties) {
    costUSD
  }
}
"""

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

    operation_name = "ExampleQuery"

    data = None

    # Make the request
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
    except Exception as err:
        print(f"Other error occurred: {err}")

    if not data:
        raise Exception("No data returned from Helicone")
    return data.get("data", {}).get("aggregatedHeliconeRequest", {}).get("cost", None)
