import json

import requests

# Define the endpoint of your GraphQL server
url = "https://www.helicone.ai/api/graphql"

# Set the headers, usually you'd need to set the content type and possibly an authorization token
headers = {"authorization": "Bearer sk-"}

# Define the query, variables, and operation name
query = """
query ExampleQuery($limit: Int, $filters: [HeliconeRequestFilter!]) {
  user {
    id
  }
  heliconeRequest(limit: $limit,filters: $filters) {
    responseBody
  }
}
"""

variables = {
    "limit": 100,
    "filters": [{"property": {"value": {"equals": "beebot"}, "name": "agent"}}],
}

operation_name = "ExampleQuery"

# Make the request
response = requests.post(
    url,
    headers=headers,
    json={"query": query, "variables": variables, "operationName": operation_name},
)
data = response.json()
total_tokens_sum = 0

for item in data["data"]["heliconeRequest"]:
    total_tokens_sum += item["responseBody"]["usage"]["total_tokens"]

# Extract the data from the response (consider adding error checks)

print(json.dumps(data, indent=4, ensure_ascii=False))
print(total_tokens_sum)
