# Joy Verify Trust Block

Verify if an agent meets a minimum trust threshold before delegation.

## Overview

Use this block as a safety gate in your workflow - only proceed with delegation if the target agent has sufficient trust score. Returns a boolean indicating whether the threshold is met.

## Inputs

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | string | The Joy agent ID to verify (e.g. 'ag_abc123') |
| `min_trust_score` | float | Minimum trust score required (0-5 scale). Default: 1.5 |
| `credentials` | API Key | Joy API key (optional, increases rate limits) |

## Outputs

| Field | Type | Description |
|-------|------|-------------|
| `meets_threshold` | boolean | True if agent's trust score meets or exceeds the minimum |
| `trust_score` | float | The agent's current trust score (0-5 scale) |
| `agent_name` | string | Name of the verified agent |
| `verified` | boolean | Whether the agent has endpoint verification |
| `error` | string | Error message if verification failed |

## Recommended Thresholds

| Level | Score | Use Case |
|-------|-------|----------|
| Permissive | 1.0 | Low-risk tasks, broad agent discovery |
| Standard | 1.5 | General use (recommended default) |
| Moderate | 2.0 | Established agents only |
| Strict | 2.5 | High security, top-tier agents |

## Example Use Cases

1. **Safety Checkpoint**: Gate a delegation workflow - only proceed if target agent is trusted
2. **Trust Auditing**: Log trust scores for compliance before every external call
3. **Conditional Routing**: Route to different agents based on trust requirements

## Links

- [Joy Trust Network](https://choosejoy.com.au)
- [API Documentation](https://choosejoy.com.au/docs)
