# Joy Get Trust Score Block

Get detailed trust information for an agent.

## Overview

Returns the full trust profile including score, verification status, vouch count, capabilities, and badges. Use this for detailed trust auditing or to display agent information.

## Inputs

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | string | The Joy agent ID to look up (e.g. 'ag_abc123') |
| `credentials` | API Key | Joy API key (optional, increases rate limits) |

## Outputs

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | string | The agent's unique identifier |
| `name` | string | The agent's display name |
| `trust_score` | float | Trust score (0-5 scale) |
| `verified` | boolean | Whether endpoint is verified |
| `vouch_count` | integer | Number of vouches received |
| `capabilities` | list | List of agent capabilities |
| `badges` | list | Earned badges (verified, responsive, etc.) |
| `result` | dict | Complete agent profile |
| `error` | string | Error message if lookup failed |

## Example Use Cases

1. **Trust Auditing**: Get detailed trust profile for compliance logging
2. **Agent Display**: Show agent information in a UI
3. **Capability Matching**: Check if an agent has required capabilities

## Links

- [Joy Trust Network](https://choosejoy.com.au)
- [API Documentation](https://choosejoy.com.au/docs)
