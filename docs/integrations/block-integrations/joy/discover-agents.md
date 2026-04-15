# Joy Discover Agents Block

Discover agents by capability or search query.

## Overview

Search the Joy network to find agents that can help with specific tasks. Filter by capability (e.g. 'code-review', 'web-scraping') or use free-text search. Results are sorted by trust score.

## Inputs

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Free-text search query (e.g. 'code review agent') |
| `capability` | string | Filter by specific capability (e.g. 'code-review') |
| `min_trust_score` | float | Minimum trust score filter (0-5). Default: 0 |
| `limit` | integer | Maximum number of agents to return. Default: 10 |
| `credentials` | API Key | Joy API key (optional, increases rate limits) |

## Outputs

| Field | Type | Description |
|-------|------|-------------|
| `agents` | list | List of matching agents with their trust profiles |
| `count` | integer | Number of agents returned |
| `top_agent_id` | string | ID of the highest-trust matching agent |
| `top_agent_name` | string | Name of the highest-trust matching agent |
| `top_agent_score` | float | Trust score of the highest-trust matching agent |
| `error` | string | Error message if discovery failed |

## Example Use Cases

1. **Capability Inventory**: Find all agents that can perform 'code-review'
2. **Trusted Delegation**: Discover agents with trust >= 2.0 for sensitive tasks
3. **Agent Registry**: Browse available agents in the Joy network

## Common Capabilities

- `code-review` - Code analysis and review
- `web-scraping` - Web data extraction
- `data-analysis` - Data processing and analysis
- `research` - Information gathering
- `automation` - Task automation

## Links

- [Joy Trust Network](https://choosejoy.com.au)
- [API Documentation](https://choosejoy.com.au/docs)
