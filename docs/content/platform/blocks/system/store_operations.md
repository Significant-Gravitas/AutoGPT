# Get Store Agent Details

### What it is
Get detailed information about an agent from the store.

### What it does
Get detailed information about an agent from the store

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Search Store Agents

### What it is
Search for agents in the store.

### What it does
Search for agents in the store

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query to find agents | str | No |
| category | Filter by category | str | No |
| sort_by | How to sort the results | "rating" | "runs" | "name" | No |
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
