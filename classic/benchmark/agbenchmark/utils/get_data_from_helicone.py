import json
import logging
import os
from typing import Optional

import requests

from agbenchmark.__main__ import BENCHMARK_START_TIME
from agbenchmark.agent_interface import HELICONE_GRAPHQL_LOGS

logger = logging.getLogger(__name__)


def get_data_from_helicone(challenge: str) -> Optional[float]:
    # Define the endpoint of your GraphQL server
    url = "https://www.helicone.ai/api/graphql"

    # Set the headers, usually you'd need to set the content type
    # and possibly an authorization token
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
        logger.debug(f"Executing Helicone query:\n{query.strip()}")
        logger.debug(f"Query variables:\n{json.dumps(variables, indent=4)}")

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
        logger.error(f"Helicone returned an HTTP error: {http_err}")
        return None
    except json.JSONDecodeError:
        raw_response = response.text  # type: ignore
        logger.error(
            f"Helicone returned an invalid JSON response: '''{raw_response}'''"
        )
        return None
    except Exception as err:
        logger.error(f"Error while trying to get data from Helicone: {err}")
        return None

    if data is None or data.get("data") is None:
        logger.error("Invalid response received from Helicone: no data")
        logger.error(f"Offending response: {response}")
        return None
    return (
        data.get("data", {}).get("aggregatedHeliconeRequest", {}).get("costUSD", None)
    )
