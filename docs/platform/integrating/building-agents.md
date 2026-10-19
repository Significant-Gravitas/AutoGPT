---
description: >-
  Create, update and version agents from code: find blocks, wire nodes and
  links, declare inputs and outputs, and handle credentials and reviews.
icon: diagram-project
---

# Build agents

An agent is a **graph**: a set of **nodes**, each an instance of a **block**, joined by **links** that carry one node's output into another node's input. The API takes the same graph JSON the visual builder saves, so an agent created through the API opens in the builder, and the reverse.

Building needs **Read Block** (to discover blocks), **Write Graph** (to save), and usually **Read Graph** and **Read Library**. To try the agent as well, add **Run Agent** and **Read Run**, and **Write Library** to remove it from your library afterwards.

{% hint style="info" %}
To start from an agent that already works, build it in the visual builder, read it with `GET /graphs/{graph_id}`, then change the JSON. It's the fastest way to learn which inputs a block needs.
{% endhint %}

## Anatomy of a graph

```json
{
  "name": "Hello from the API",
  "description": "Greets whoever you name.",
  "nodes": [
    {
      "id": "name-input",
      "block_id": "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
      "input_default": { "name": "name", "title": "Your name", "value": "World" }
    },
    {
      "id": "greeting-template",
      "block_id": "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
      "input_default": { "format": "Hello, {{ name }}!", "values": {} }
    },
    {
      "id": "greeting-output",
      "block_id": "363ae599-353e-4804-937e-b2ee3cef3da4",
      "input_default": { "name": "greeting", "title": "Greeting" }
    }
  ],
  "links": [
    { "id": "name-to-template", "source_id": "name-input", "source_name": "result",
      "sink_id": "greeting-template", "sink_name": "values_#_name" },
    { "id": "template-to-output", "source_id": "greeting-template", "source_name": "output",
      "sink_id": "greeting-output", "sink_name": "value" }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `nodes[].id` | Any string unique within the request. Links refer to nodes by it. The API replaces it with a generated UUID when it saves the graph. |
| `nodes[].block_id` | Which block this node runs. Find IDs with `GET /blocks`. |
| `nodes[].input_default` | Fixed values for the block's inputs. Keys are properties of the block's `input_schema`. Inputs fed by a link can be left out. |
| `nodes[].metadata` | Optional. The builder stores the node's position here (`{"position": {"x": 0, "y": 0}}`). |
| `links[].source_id`, `source_name` | The node and output pin the data comes from. Pins are the properties of the block's `output_schema`. |
| `links[].sink_id`, `sink_name` | The node and input pin it goes to. Pins are the properties of the block's `input_schema`. |
| `links[].is_static` | Optional. A static link re-delivers its last value to every later execution of the sink, instead of delivering each value once. Links from input blocks are always static. |

## Find blocks

`GET /blocks` lists every block on the instance, about 430 on a current release, so read all the pages. Each block has an `id`, a `name`, a `description`, `categories`, an `input_schema` and an `output_schema` (both JSON Schema), a `block_type` and its `costs`.

```python
def list_blocks() -> list[dict]:
    blocks, params = [], {"limit": 100}
    while True:
        response = requests.get(f"{API_URL}/blocks", headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        page = response.json()
        blocks += page["items"]
        if not page["next_cursor"]:
            return blocks
        params["cursor"] = page["next_cursor"]


def find_blocks(blocks: list[dict], *words: str) -> list[dict]:
    return [
        block
        for block in blocks
        if all(word.casefold() in f"{block['name']} {block['description']}".casefold() for word in words)
    ]


blocks = list_blocks()
for block in find_blocks(blocks, "word", "count"):
    print(block["id"], block["name"], "-", block["description"])
```

A block's `name` is its class name, such as `WordCharacterCountBlock`. The [block reference](https://agpt.co/docs/integrations) describes each block under a spaced-out title without the suffix, such as **Word Character Count**, so drop `Block` and add the spaces to match the two. The block's `input_schema` and `output_schema` from `GET /blocks` are what the API checks your graph against.

`GET /search?query=count%20words&content_types=BLOCK` finds blocks by meaning, using the instance's search index. A new self-hosted instance usually hasn't built that index, so expect to filter the full list there.

## Inputs and outputs

What a graph takes and returns is defined by its input and output blocks. The API builds the graph's `input_schema` and `output_schema` from them.

| Block | `block_id` | Use it for |
| --- | --- | --- |
| `AgentInputBlock` | `c0a8e994-ebf1-4a9c-a4d8-89d09c86741b` | Any input. |
| `AgentShortTextInputBlock` | `7fcd3bcb-8e1b-4e69-903d-32d3d4a92158` | A line of text. |
| `AgentLongTextInputBlock` | `90a56ffb-7024-4b2b-ab50-e26c5e5ab8ba` | A paragraph or more. |
| `AgentNumberInputBlock` | `96dae2bb-97a2-41c2-bd2f-13a3b5a8ea98` | A whole number. A decimal is truncated (`2.5` becomes `2`), not rejected, so use `AgentInputBlock` for decimals. |
| `AgentToggleInputBlock` | `cbf36ab5-df4a-43b6-8a7f-f7ed8652116e` | A true/false switch. |
| `AgentDropdownInputBlock` | `655d6fdf-a334-421c-b733-520549c07cd1` | One of a fixed set of options. |
| `AgentDateInputBlock` | `7e198b09-4994-47db-8b4d-952d98241817` | A date. |
| `AgentTimeInputBlock` | `2a1c757e-86cf-4c7e-aacf-060dc382e434` | A time of day. |
| `AgentFileInputBlock` | `95ead23f-8283-4654-aef3-10c053b74a31` | A file, passed as a `workspace://` reference. |
| `AgentTableInputBlock` | `5603b273-f41e-4020-af7d-fbc9c6a8d928` | Rows of data. |
| `AgentOutputBlock` | `363ae599-353e-4804-937e-b2ee3cef3da4` | A result of the run. |

Every input block takes `name` (the key callers use in `inputs`), and optionally `title`, `description` and `value` (the default). An input with a default becomes optional. Each input block sends what it receives out of its `result` pin. A run started through the REST API without an input that has no default still starts; that input sends nothing, so the steps after it don't run.

An output block takes `name` (the key in the run's `outputs`) and receives its data on its `value` pin. A run's `outputs` record exactly the value that arrives on that pin, so format text with a block such as `FillTextTemplateBlock` before it reaches the output block. The output block's own `format` field doesn't change what the API returns.

`FillTextTemplateBlock` is the usual way to shape text. Its `format` is a [Jinja2](https://jinja.palletsprojects.com/en/stable/templates/) template rendered in a sandbox, so filters, conditions and loops work: `{{ ticket | upper }}`, `{{ tags | join(", ") }}`, `{% if urgent %}URGENT: {% endif %}`. Some changes, such as upper-casing text, have no block of their own. Values go in as they are; set `escape_html` to `true` when the result is HTML.

## Dict, list and object pins

Some inputs are a dictionary, a list or an object, and a link can fill a single entry of one. Append a separator and the key to the pin name:

| To fill | Write `sink_name` as | Example |
| --- | --- | --- |
| A key of a dictionary input | `<pin>_#_<key>` | `values_#_name` sets `values["name"]` |
| An item of a list input | `<pin>_$_<index>` | `items_$_0` sets `items[0]` |
| An attribute of an object input | `<pin>_@_<attribute>` | `payload_@_title` sets `payload.title` |

The quickstart's agent uses `values_#_name` to put the input into the template's `values` dictionary, which the template then reads as `{{ name }}`.

## Create the agent

```bash
curl -s -X POST "$AUTOGPT_API_URL/graphs" \
  -H "X-API-Key: $AUTOGPT_API_KEY" -H "Content-Type: application/json" \
  --data @hello_agent.json
```

The API validates the graph before saving it. On success it returns `201` with the saved graph: generated node IDs, `"version": 1`, `"is_active": true`, and the computed `input_schema`, `output_schema` and `credentials_input_schema`. The graph is added to your library automatically; get the library agent to run it with `GET /graphs/{graph_id}/library-agent`.

An invalid graph fails with `400` and a message naming the problem. Nodes in the message carry the IDs the API generated, not the ones you sent:

```json
{
  "error": {
    "code": "bad_request",
    "message": "Link ('99299fca-...', 'no_such_pin') <-> ('4685959d-...', 'value'), `no_such_pin` invalid, Allowed fields: {'result'}",
    "details": null
  }
}
```

Common causes: a `block_id` that doesn't exist on this instance, a pin name that isn't in the block's schema, or a link to a node ID that isn't in `nodes`.

## Credentials

A block that calls a third-party service declares a credentials input. You don't put credentials in the graph. The graph's `credentials_input_schema` lists what the agent needs, and each run supplies them in `credentials_inputs`. See [Supply credentials it needs](running-agents.md#supply-credentials-it-needs).

## Update an agent

Graphs are versioned, and a saved version never changes. `PUT /graphs/{graph_id}` with a complete graph definition, in the same shape as for `POST /graphs`, saves it as the next version. The body needs no `id` or `version`:

```bash
curl -s -X PUT "$AUTOGPT_API_URL/graphs/$GRAPH_ID" \
  -H "X-API-Key: $AUTOGPT_API_KEY" -H "Content-Type: application/json" \
  --data @hello_agent_v2.json
```

* The new version becomes the **active** one unless you send `"is_active": false`, and your library agent moves to it, keeping its ID.
* Every `PUT` saves a new version, even when nothing changed. A deploy script should skip the call when its graph JSON is the same as last time.
* To update an agent you made earlier, [find it by name](running-agents.md#find-the-agent) and use its `graph_id`.
* `GET /graphs/{graph_id}/versions` lists every version. `GET /graphs/{graph_id}?version=2` reads one.
* `PUT /graphs/{graph_id}/versions/active` with `{"active_graph_version": 1}` rolls back.
* A run uses the version its library agent points at. `PATCH /library/agents/{agent_id}` with `{"graph_version": 1}` pins it to a version, and `{"auto_update_version": true}` makes it follow the active one.

## Human-in-the-loop blocks

The `HumanInTheLoopBlock` (`8b2a7b3c-6e9d-4a5f-8c1b-2e3f4a5b6c7d`) pauses the run until someone reviews its `data`. Its `name` input becomes the review's `instructions`. Connect `approved_data` to the steps that should follow an approval and `rejected_data` to the steps that should follow a rejection. Runs pause in `REVIEW` until [the review is answered](running-agents.md#human-in-the-loop-reviews).

To let such an agent run unattended, turn reviews off for the graph, and every review is approved automatically:

```bash
curl -s -X PATCH "$AUTOGPT_API_URL/graphs/$GRAPH_ID/settings" \
  -H "X-API-Key: $AUTOGPT_API_KEY" -H "Content-Type: application/json" \
  -d '{"human_in_the_loop_safe_mode": false}'
```

## Copy an existing agent

* `POST /library/agents/{agent_id}/fork` copies one of your library agents into a new, independent graph you can change (needs **Write Library**).
* `POST /marketplace/agents/{username}/{agent_name}/add-to-library` adds a marketplace agent to your library. Fork it to edit it.

## Let an AI build it

The [MCP server](mcp-server.md) has tools that turn a plain-language description into a working agent. `get_agent_building_guide` returns the building guide, `create_agent` and `edit_agent` validate, fix and save graph JSON, and `validate_agent_graph` and `fix_agent_graph` check graph JSON without saving it.
