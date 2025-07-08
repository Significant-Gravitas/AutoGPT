
Exa home pagelight logo

Search...
⌘K

Ask AI
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
FAQ
Dashboard
Get started
Example queries
Import from CSV
Exclude Results
Integrations
Videos
API
Overview
Get started
How It Works
Core
Imports
Monitors
Webhooks
POST
Create a Webhook
GET
Get a Webhook
PATCH
Update a Webhook
DEL
Delete a Webhook
GET
List webhooks
GET
List webhook attempts
Verifying Signatures
Events
Webhooks
Create a Webhook
POST
/
v0
/
webhooks

Try it
Authorizations
​
x-api-key
stringheaderrequired
Your Exa API key
Body
application/json
​
events
enum<string>[]required
The events to trigger the webhook
Required array length: 1 - 15 elements
Show child attributes
​
url
stringrequired
The URL to send the webhook to
​
metadata
object
Set of key-value pairs you want to associate with this object.
Show child attributes
Response
200 - application/json
Webhook
​
id
stringrequired
The unique identifier for the webhook
​
object
stringdefault:webhookrequired
Allowed value: "webhook"
​
status
enum<string>required
The status of the webhook
Available options: active, inactive 
​
events
enum<string>[]required
The events to trigger the webhook
Minimum length: 1
Show child attributes
​
url
stringrequired
The URL to send the webhook to
​
secret
string | nullrequired
The secret to verify the webhook signature. Only returned on Webhook creation.
​
createdAt
stringrequired
The date and time the webhook was created
​
updatedAt
stringrequired
The date and time the webhook was last updated
​
metadata
object
The metadata of the webhook
Show child attributes
Get Monitor Run
Get a Webhook
x
discord
Powered by Mintlify

cURL

Python

JavaScript

Copy

Ask AI
# pip install exa-py
from exa_py import Exa
exa = Exa('YOUR_EXA_API_KEY')

webhook = exa.websets.webhooks.create(params={
    'url': 'https://api.yourapp.com/webhooks/exa',
    'events': ['webset.completed', 'enrichment.completed']
})

print(f'Created webhook: {webhook.id}')

200

Copy

Ask AI
{
  "id": "<string>",
  "object": "webhook",
  "status": "active",
  "events": [
    "webset.created"
  ],
  "url": "<string>",
  "secret": "<string>",
  "metadata": {},
  "createdAt": "2023-11-07T05:31:56Z",
  "updatedAt": "2023-11-07T05:31:56Z"
}
Assistant


Responses are generated using AI and may contain mistakes.

Create a Webhook - Exa


Exa home pagelight logo

Search...
⌘K

Ask AI
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
FAQ
Dashboard
Get started
Example queries
Import from CSV
Exclude Results
Integrations
Videos
API
Overview
Get started
How It Works
Core
Imports
Monitors
Webhooks
POST
Create a Webhook
GET
Get a Webhook
PATCH
Update a Webhook
DEL
Delete a Webhook
GET
List webhooks
GET
List webhook attempts
Verifying Signatures
Events
Webhooks
Get a Webhook
GET
/
v0
/
webhooks
/
{id}

Try it
Authorizations
​
x-api-key
stringheaderrequired
Your Exa API key
Path Parameters
​
id
stringrequired
The id of the webhook
Response
200

application/json
Webhook
​
id
stringrequired
The unique identifier for the webhook
​
object
stringdefault:webhookrequired
Allowed value: "webhook"
​
status
enum<string>required
The status of the webhook
Available options: active, inactive 
​
events
enum<string>[]required
The events to trigger the webhook
Minimum length: 1
Show child attributes
​
url
stringrequired
The URL to send the webhook to
​
secret
string | nullrequired
The secret to verify the webhook signature. Only returned on Webhook creation.
​
createdAt
stringrequired
The date and time the webhook was created
​
updatedAt
stringrequired
The date and time the webhook was last updated
​
metadata
object
The metadata of the webhook
Show child attributes
Create a Webhook
Update a Webhook
x
discord
Powered by Mintlify

cURL

Python

JavaScript

Copy

Ask AI
# pip install exa-py
from exa_py import Exa
exa = Exa('YOUR_EXA_API_KEY')

webhook = exa.websets.webhooks.get('webhook_id')

print(f'Webhook: {webhook.id} - {webhook.url}')

200

404

Copy

Ask AI
{
  "id": "<string>",
  "object": "webhook",
  "status": "active",
  "events": [
    "webset.created"
  ],
  "url": "<string>",
  "secret": "<string>",
  "metadata": {},
  "createdAt": "2023-11-07T05:31:56Z",
  "updatedAt": "2023-11-07T05:31:56Z"
}
Assistant


Responses are generated using AI and may contain mistakes.

Get a Webhook - Exa


Exa home pagelight logo

Search...
⌘K

Ask AI
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
FAQ
Dashboard
Get started
Example queries
Import from CSV
Exclude Results
Integrations
Videos
API
Overview
Get started
How It Works
Core
Imports
Monitors
Webhooks
POST
Create a Webhook
GET
Get a Webhook
PATCH
Update a Webhook
DEL
Delete a Webhook
GET
List webhooks
GET
List webhook attempts
Verifying Signatures
Events
Webhooks
Update a Webhook
PATCH
/
v0
/
webhooks
/
{id}

Try it
Authorizations
​
x-api-key
stringheaderrequired
Your Exa API key
Path Parameters
​
id
stringrequired
The id of the webhook
Body
application/json
​
events
enum<string>[]
The events to trigger the webhook
Required array length: 1 - 15 elements
Show child attributes
​
url
string
The URL to send the webhook to
​
metadata
object
Set of key-value pairs you want to associate with this object.
Show child attributes
Response
200

application/json
Webhook
​
id
stringrequired
The unique identifier for the webhook
​
object
stringdefault:webhookrequired
Allowed value: "webhook"
​
status
enum<string>required
The status of the webhook
Available options: active, inactive 
​
events
enum<string>[]required
The events to trigger the webhook
Minimum length: 1
Show child attributes
​
url
stringrequired
The URL to send the webhook to
​
secret
string | nullrequired
The secret to verify the webhook signature. Only returned on Webhook creation.
​
createdAt
stringrequired
The date and time the webhook was created
​
updatedAt
stringrequired
The date and time the webhook was last updated
​
metadata
object
The metadata of the webhook
Show child attributes
Get a Webhook
Delete a Webhook
x
discord
Powered by Mintlify

cURL

Python

JavaScript

Copy

Ask AI
# pip install exa-py
from exa_py import Exa
exa = Exa('YOUR_EXA_API_KEY')

webhook = exa.websets.webhooks.update('webhook_id', params={
    'url': 'https://api.yourapp.com/webhooks/exa-updated',
    'events': ['webset.completed']
})

print(f'Updated webhook: {webhook.id}')

200

404

Copy

Ask AI
{
  "id": "<string>",
  "object": "webhook",
  "status": "active",
  "events": [
    "webset.created"
  ],
  "url": "<string>",
  "secret": "<string>",
  "metadata": {},
  "createdAt": "2023-11-07T05:31:56Z",
  "updatedAt": "2023-11-07T05:31:56Z"
}
Assistant


Responses are generated using AI and may contain mistakes.

Update a Webhook - Exa


Exa home pagelight logo

Search...
⌘K

Ask AI
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
FAQ
Dashboard
Get started
Example queries
Import from CSV
Exclude Results
Integrations
Videos
API
Overview
Get started
How It Works
Core
Imports
Monitors
Webhooks
POST
Create a Webhook
GET
Get a Webhook
PATCH
Update a Webhook
DEL
Delete a Webhook
GET
List webhooks
GET
List webhook attempts
Verifying Signatures
Events
Webhooks
Delete a Webhook
DELETE
/
v0
/
webhooks
/
{id}

Try it
Authorizations
​
x-api-key
stringheaderrequired
Your Exa API key
Path Parameters
​
id
stringrequired
The id of the webhook
Response
200

application/json
Webhook
​
id
stringrequired
The unique identifier for the webhook
​
object
stringdefault:webhookrequired
Allowed value: "webhook"
​
status
enum<string>required
The status of the webhook
Available options: active, inactive 
​
events
enum<string>[]required
The events to trigger the webhook
Minimum length: 1
Show child attributes
​
url
stringrequired
The URL to send the webhook to
​
secret
string | nullrequired
The secret to verify the webhook signature. Only returned on Webhook creation.
​
createdAt
stringrequired
The date and time the webhook was created
​
updatedAt
stringrequired
The date and time the webhook was last updated
​
metadata
object
The metadata of the webhook
Show child attributes
Update a Webhook
List webhooks
x
discord
Powered by Mintlify

cURL

Python

JavaScript

Copy

Ask AI
# pip install exa-py
from exa_py import Exa
exa = Exa('YOUR_EXA_API_KEY')

exa.websets.webhooks.delete('webhook_id')

print('Webhook deleted successfully')

200

404

Copy

Ask AI
{
  "id": "<string>",
  "object": "webhook",
  "status": "active",
  "events": [
    "webset.created"
  ],
  "url": "<string>",
  "secret": "<string>",
  "metadata": {},
  "createdAt": "2023-11-07T05:31:56Z",
  "updatedAt": "2023-11-07T05:31:56Z"
}
Assistant


Responses are generated using AI and may contain mistakes.

Delete a Webhook - Exa



Exa home pagelight logo

Search...
⌘K

Ask AI
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
FAQ
Dashboard
Get started
Example queries
Import from CSV
Exclude Results
Integrations
Videos
API
Overview
Get started
How It Works
Core
Imports
Monitors
Webhooks
POST
Create a Webhook
GET
Get a Webhook
PATCH
Update a Webhook
DEL
Delete a Webhook
GET
List webhooks
GET
List webhook attempts
Verifying Signatures
Events
Webhooks
List webhooks
GET
/
v0
/
webhooks

Try it
Authorizations
​
x-api-key
stringheaderrequired
Your Exa API key
Query Parameters
​
cursor
string
The cursor to paginate through the results
Minimum length: 1
​
limit
numberdefault:25
The number of results to return
Required range: 1 <= x <= 200
Response
200 - application/json
List of webhooks
​
data
object[]required
The list of webhooks
Show child attributes
​
hasMore
booleanrequired
Whether there are more results to paginate through
​
nextCursor
string | nullrequired
The cursor to paginate through the next set of results
Delete a Webhook
List webhook attempts
x
discord
Powered by Mintlify

cURL

Python

JavaScript

Copy

Ask AI
# pip install exa-py
from exa_py import Exa
exa = Exa('YOUR_EXA_API_KEY')

webhooks = exa.websets.webhooks.list()

print(f'Found {len(webhooks.data)} webhooks')
for webhook in webhooks.data:
    print(f'- {webhook.id}: {webhook.url}')

200

Copy

Ask AI
{
  "data": [
    {
      "id": "<string>",
      "object": "webhook",
      "status": "active",
      "events": [
        "webset.created"
      ],
      "url": "<string>",
      "secret": "<string>",
      "metadata": {},
      "createdAt": "2023-11-07T05:31:56Z",
      "updatedAt": "2023-11-07T05:31:56Z"
    }
  ],
  "hasMore": true,
  "nextCursor": "<string>"
}
Assistant


Responses are generated using AI and may contain mistakes.

List webhooks - Exa



Exa home pagelight logo

Search...
⌘K

Ask AI
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
FAQ
Dashboard
Get started
Example queries
Import from CSV
Exclude Results
Integrations
Videos
API
Overview
Get started
How It Works
Core
Imports
Monitors
Webhooks
POST
Create a Webhook
GET
Get a Webhook
PATCH
Update a Webhook
DEL
Delete a Webhook
GET
List webhooks
GET
List webhook attempts
Verifying Signatures
Events
Webhooks
List webhook attempts
List all attempts made by a Webhook ordered in descending order.
GET
/
v0
/
webhooks
/
{id}
/
attempts

Try it
Authorizations
​
x-api-key
stringheaderrequired
Your Exa API key
Path Parameters
​
id
stringrequired
The ID of the webhook
Query Parameters
​
cursor
string
The cursor to paginate through the results
Minimum length: 1
​
limit
numberdefault:25
The number of results to return
Required range: 1 <= x <= 200
​
eventType
enum<string>
The type of event to filter by
Available options: webset.created, webset.deleted, webset.paused, webset.idle, webset.search.created, webset.search.canceled, webset.search.completed, webset.search.updated, import.created, import.completed, import.processing, webset.item.created, webset.item.enriched, webset.export.created, webset.export.completed 
Response
200 - application/json
List of webhook attempts
​
data
object[]required
The list of webhook attempts
Show child attributes
​
hasMore
booleanrequired
Whether there are more results to paginate through
​
nextCursor
string | nullrequired
The cursor to paginate through the next set of results
List webhooks
Verifying Signatures
x
discord
Powered by Mintlify

cURL

Python

JavaScript

Copy

Ask AI
# pip install exa-py
from exa_py import Exa
exa = Exa('YOUR_EXA_API_KEY')

attempts = exa.websets.webhooks.attempts.list('webhook_id')

print(f'Found {len(attempts.data)} webhook attempts')
for attempt in attempts.data:
    print(f'- {attempt.id}: {attempt.status}')

200

Copy

Ask AI
{
  "data": [
    {
      "id": "<string>",
      "object": "webhook_attempt",
      "eventId": "<string>",
      "eventType": "webset.created",
      "webhookId": "<string>",
      "url": "<string>",
      "successful": true,
      "responseHeaders": {},
      "responseBody": "<string>",
      "responseStatusCode": 123,
      "attempt": 123,
      "attemptedAt": "2023-11-07T05:31:56Z"
    }
  ],
  "hasMore": true,
  "nextCursor": "<string>"
}
Assistant


Responses are generated using AI and may contain mistakes.

List webhook attempts - Exa