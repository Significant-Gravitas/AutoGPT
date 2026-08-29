# System Library Operations
<!-- MANUAL: file_description -->
Blocks for managing agents in your personal AutoGPT library.
<!-- END MANUAL -->

## Add To Library From Store

### What it is
Add an agent from the store to your personal library

### How it works
<!-- MANUAL: how_it_works -->
This block copies an agent from the public store into your personal library using its store_listing_version_id. Optionally provide a custom agent_name to rename it in your library.

The block returns the library entry ID and agent graph ID, which can be used to execute the agent or manage your library.
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
**Agent Provisioning**: Automatically add recommended agents to a user's library.

**Onboarding Flows**: Set up a user's library with starter agents during onboarding.

**Dynamic Agent Access**: Add agents on-demand when users request specific capabilities.
<!-- END MANUAL -->

---

## List Library Agents

### What it is
List all agents in your personal library

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all agents stored in your personal library. Use search_query to filter by name, and limit/page for pagination through large libraries.

Results include each agent's metadata and are output both as a complete list and individually for iteration.
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
**Agent Selection**: Display available agents for users to choose from in a workflow.

**Library Management**: Build interfaces for users to manage and organize their agent library.

**Agent Inventory**: Check what agents are available before deciding which to execute.
<!-- END MANUAL -->

---
