# An example to get the remaining rate limit using the Github GraphQL API.
import os 
import requests
access_token = os.environ.get("GH_TOKEN")
#headers = {"Authorization": "Bearer YOUR API KEY"}
import logging

# # These two lines enable debugging at httplib level (requests->urllib3->http.client)
# # You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
# # The only thing missing will be the response.body which is not logged.
# try:
#     import http.client as http_client
# except ImportError:
#     # Python 2
#     import httplib as http_client
# http_client.HTTPConnection.debuglevel = 1
# # You must initialize logging, otherwise you'll not see debug output.
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True

headers = {
       'Authorization': f'Bearer {access_token}',
    }

def run_query(query): # A simple function to use requests.post to make the API call. Note the json= section.
    request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

        
# The GraphQL query (with a few aditional bits included) itself defined as a multi-line string.

page = 1
has_next_page = True
after_cursor = None

while has_next_page:
    #print("step")
    after = ""
    if after_cursor:
        after = f", after: \"{after_cursor}\"" 
    query = f"""{{  repository(owner: "Significant-Gravitas", name: "AutoGPT") {{
        forks(first: 100 {after}) {{
          pageInfo {{
            endCursor
            hasNextPage
          }}
          nodes {{
            nameWithOwner
          }}
        }}
      }}
    }}
    """
    #print(query)
    result = run_query(query) # Execute the query

    page= result["data"]["repository"]["forks"]["pageInfo"]
    nodes= result["data"]["repository"]["forks"]["nodes"]
    has_next_page = page["hasNextPage"    ]
    after_cursor = page["endCursor"]    
    for x in  nodes:
        print(x)

