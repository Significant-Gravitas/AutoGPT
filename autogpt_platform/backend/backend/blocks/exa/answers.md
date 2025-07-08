

Exa home pagelight logo

Search or ask...
⌘K
Exa Search
Log In
API Dashboard
Documentation
Examples
Integrations
SDKs
Websets
Changelog
Discord
Blog
Getting Started

Overview
Quickstart
API Reference

POST
Search
POST
Get contents
POST
Find similar links
POST
Answer
OpenAPI Specification
RAG Quick Start Guide

RAG with Exa and OpenAI
RAG with LangChain
OpenAI Exa Wrapper
CrewAI agents with Exa
RAG with LlamaIndex
Tool calling with GPT
Tool calling with Claude
OpenAI Chat Completions
OpenAI Responses API
Concepts

How Exa Search Works
The Exa Index
Contents retrieval with Exa API
Exa's Capabilities Explained
FAQs
Crawling Subpages with Exa
Exa LiveCrawl
Admin

Setting Up and Managing Your Team
Rate Limits
Enterprise Documentation & Security
API Reference
Answer
Get an LLM answer to a question informed by Exa search results. Fully compatible with OpenAI’s chat completions endpoint - docs here. /answer performs an Exa search and uses an LLM to generate either:

A direct answer for specific queries. (i.e. “What is the capital of France?” would return “Paris”)
A detailed summary with citations for open-ended queries (i.e. “What is the state of ai in healthcare?” would return a summary with citations to relevant sources)
The response includes both the generated answer and the sources used to create it. The endpoint also supports streaming (as stream=True), which will returns tokens as they are generated.
POST
/
answer

Try it
Get your Exa API key

Authorizations
​
x-api-key
stringheaderrequired
API key can be provided either via x-api-key header or Authorization header with Bearer scheme
Body
application/json
​
query
stringrequired
The question or query to answer.
Minimum length: 1
Example:
"What is the latest valuation of SpaceX?"
​
stream
booleandefault:false
If true, the response is returned as a server-sent events (SSS) stream.
​
text
booleandefault:false
If true, the response includes full text content in the search results
​
model
enum<string>default:exa
The search model to use for the answer. Exa passes only one query to exa, while exa-pro also passes 2 expanded queries to our search model.
Available options: exa, exa-pro 
Response
200
application/json

OK
​
answer
string
The generated answer based on search results.
Example:
"$350 billion."
​
citations
object[]
Search results used to generate the answer.

Show child attributes
​
costDollars
object

Show child attributes
Find similar links
OpenAPI Specification
x
discord
Powered by Mintlify

cURL

Python

JavaScript

Copy
# pip install exa-py
from exa_py import Exa
exa = Exa('YOUR_EXA_API_KEY')

result = exa.answer(
    "What is the latest valuation of SpaceX?",
    text=True
)

print(result)

200

Copy
{
  "answer": "$350 billion.",
  "citations": [
    {
      "id": "https://www.theguardian.com/science/2024/dec/11/spacex-valued-at-350bn-as-company-agrees-to-buy-shares-from-employees",
      "url": "https://www.theguardian.com/science/2024/dec/11/spacex-valued-at-350bn-as-company-agrees-to-buy-shares-from-employees",
      "title": "SpaceX valued at $350bn as company agrees to buy shares from ...",
      "author": "Dan Milmon",
      "publishedDate": "2023-11-16T01:36:32.547Z",
      "text": "SpaceX valued at $350bn as company agrees to buy shares from ...",
      "image": "https://i.guim.co.uk/img/media/7cfee7e84b24b73c97a079c402642a333ad31e77/0_380_6176_3706/master/6176.jpg?width=1200&height=630&quality=85&auto=format&fit=crop&overlay-align=bottom%2Cleft&overlay-width=100p&overlay-base64=L2ltZy9zdGF0aWMvb3ZlcmxheXMvdGctZGVmYXVsdC5wbmc&enable=upscale&s=71ebb2fbf458c185229d02d380c01530",
      "favicon": "https://assets.guim.co.uk/static/frontend/icons/homescreen/apple-touch-icon.svg"
    }
  ],
  "costDollars": {
    "total": 0.005,
    "breakDown": [
      {
        "search": 0.005,
        "contents": 0,
        "breakdown": {
          "keywordSearch": 0,
          "neuralSearch": 0.005,
          "contentText": 0,
          "contentHighlight": 0,
          "contentSummary": 0
        }
      }
    ],
    "perRequestPrices": {
      "neuralSearch_1_25_results": 0.005,
      "neuralSearch_26_100_results": 0.025,
      "neuralSearch_100_plus_results": 1,
      "keywordSearch_1_100_results": 0.0025,
      "keywordSearch_100_plus_results": 3
    },
    "perPagePrices": {
      "contentText": 0.001,
      "contentHighlight": 0.001,
      "contentSummary": 0.001
    }
  }
}
