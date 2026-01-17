# System Store Operations
<!-- MANUAL: file_description -->
Blocks for browsing and retrieving agent details from the AutoGPT store.
<!-- END MANUAL -->

## Get Store Agent Details

### What it is
Get detailed information about an agent from the store

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves detailed metadata about a specific agent from the AutoGPT store using the creator's username and agent slug. It returns the agent's name, description, categories, run count, and average rating.

The store_listing_version_id can be used with other blocks to add the agent to your library or execute it.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| creator | The username of the agent creator | str | Yes |
| slug | The name of the agent | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| found | Whether the agent was found in the store | bool |
| store_listing_version_id | The store listing version ID | str |
| agent_name | Name of the agent | str |
| description | Description of the agent | str |
| creator | Creator of the agent | str |
| categories | Categories the agent belongs to | List[str] |
| runs | Number of times the agent has been run | int |
| rating | Average rating of the agent | float |

### Possible use case
<!-- MANUAL: use_case -->
**Agent Discovery**: Fetch details about a specific agent before adding it to your library.

**Agent Validation**: Check an agent's ratings and run count to assess quality and popularity.

**Dynamic Agent Selection**: Get agent metadata to decide which version or variant to use.
<!-- END MANUAL -->

---

## Search Store Agents

### What it is
Search for agents in the store

### How it works
<!-- MANUAL: how_it_works -->
This block searches the AutoGPT agent store using a query string. Filter results by category and sort by rating, runs, or name. Limit controls the maximum number of results returned.

Results include basic agent information and are output both as a list and individually for workflow iteration.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query to find agents | str | No |
| category | Filter by category | str | No |
| sort_by | How to sort the results | "rating" \| "runs" \| "name" \| "updated_at" | No |
| limit | Maximum number of results to return | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| agents | List of agents matching the search criteria | List[StoreAgent] |
| agent | Basic information of the agent | StoreAgent |
| total_count | Total number of agents found | int |

### Possible use case
<!-- MANUAL: use_case -->
**Agent Recommendation**: Search for agents that match user needs and recommend the best options.

**Marketplace Browse**: Allow users to explore available agents by category or keyword.

**Agent Orchestration**: Find and compose multiple specialized agents for complex workflows.
<!-- END MANUAL -->

---
