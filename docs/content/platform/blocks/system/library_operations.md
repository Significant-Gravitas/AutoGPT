# Add To Library From Store

### What it is
Add an agent from the store to your personal library.

### What it does
Add an agent from the store to your personal library

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| store_listing_version_id | The ID of the store listing version to add to library | str | Yes |
| agent_name | Optional custom name for the agent in your library | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the agent was successfully added to library | bool |
| library_agent_id | The ID of the library agent entry | str |
| agent_id | The ID of the agent graph | str |
| agent_version | The version number of the agent graph | int |
| agent_name | The name of the agent | str |
| message | Success or error message | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## List Library Agents

### What it is
List all agents in your personal library.

### What it does
List all agents in your personal library

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| search_query | Optional search query to filter agents | str | No |
| limit | Maximum number of agents to return | int | No |
| page | Page number for pagination | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| agents | List of agents in the library | List[LibraryAgent] |
| agent | Individual library agent (yielded for each agent) | LibraryAgent |
| total_count | Total number of agents in library | int |
| page | Current page number | int |
| total_pages | Total number of pages | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
