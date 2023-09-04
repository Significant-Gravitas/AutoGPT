from gql.transport.aiohttp import AIOHTTPTransport
from gql import gql, Client
import os

helicone_api_key = os.getenv('HELICONE_API_KEY')

url = "https://www.helicone.ai/api/graphql"
# Replace <KEY> with your personal access key
transport = AIOHTTPTransport(url=url, headers={
    "authorization": f"Bearer {helicone_api_key}"
})

client = Client(transport=transport, fetch_schema_from_transport=True)

MAX_LOOPS = 10
SIZE = 100
import pandas as pd

data = []

for i in range(MAX_LOOPS):
    query = gql(
        """
        query ExampleQuery($limit: Int, $offset: Int){
            heliconeRequest(
                limit: $limit
                offset: $offset
            ) {
                prompt
                properties{
                    name
                    value
                }
                
                requestBody
                response
                createdAt

            }

            }
    """
    )
    result = client.execute(query,
                            variable_values={
                                "limit": SIZE,
                                "offset": i * SIZE
                            }
                            )

    for item in result["heliconeRequest"]:
        properties = {prop['name']: prop['value'] for prop in item['properties']}
        data.append({
            'createdAt': item['createdAt'],
            'agent': properties.get('agent'),
            'job_id': properties.get('job_id'),
            'challenge': properties.get('challenge'),
            'benchmark_start_time': properties.get('benchmark_start_time'),
            'prompt': item['prompt'],
            'model': item['requestBody'].get('model'),
            'request': item['requestBody'].get('messages'),
        })

    if (len(result["heliconeRequest"]) == 0):
        print("No more results")
        break

df = pd.DataFrame(data)
import IPython
IPython.embed()
print(df.head())