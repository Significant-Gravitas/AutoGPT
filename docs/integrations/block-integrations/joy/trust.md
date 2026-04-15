# Joy Trust
<!-- MANUAL: file_description -->
Verify agent trust scores before delegating tasks. Joy Trust Network provides cross-platform reputation scoring for AI agents based on vouches, verification, and behavioral signals.
<!-- END MANUAL -->

## Joy Get Trust Score

### What it is
Get detailed trust profile for an agent including score, verification status, and capabilities.

### How it works
<!-- MANUAL: how_it_works -->
The block queries the Joy API for a complete agent profile including trust score, verification status, vouch count, capabilities, and badges.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| agent_id | The Joy agent ID to look up (e.g. 'ag_abc123') | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if lookup failed | str |
| agent_id | The agent's unique identifier | str |
| name | The agent's display name | str |
| trust_score | Trust score (0-5 scale) | float |
| verified | Whether endpoint is verified | bool |
| vouch_count | Number of vouches received | int |
| capabilities | List of agent capabilities | List[Any] |
| badges | Earned badges (verified, responsive, etc.) | List[Any] |
| result | Complete agent profile | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Agent Directory UI** — Display agent cards with trust scores and badges.
**Capability Matching** — Check if an agent has required capabilities before delegation.
<!-- END MANUAL -->

---

## Joy Verify Trust

### What it is
Verify if an agent meets minimum trust threshold before delegating tasks. Use as a safety gate in multi-agent workflows.

### How it works
<!-- MANUAL: how_it_works -->
The block calls the Joy API to retrieve an agent's trust profile by ID. It compares the agent's trust score against your specified minimum threshold and returns a boolean indicating whether the agent is sufficiently trusted.

Recommended thresholds:
- **1.0 (Permissive):** Low-risk tasks
- **1.5 (Standard):** General use, recommended default
- **2.0 (Moderate):** Established agents only
- **2.5 (Strict):** High security
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| agent_id | The Joy agent ID to verify (e.g. 'ag_abc123') | str | Yes |
| min_trust_score | Minimum trust score required (0-5 scale). Default 1.5 is recommended for general use. | float | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if verification failed | str |
| meets_threshold | True if agent's trust score meets or exceeds the minimum threshold | bool |
| trust_score | The agent's current trust score (0-5 scale) | float |
| agent_name | Name of the verified agent | str |
| verified | Whether the agent has endpoint verification | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Safety Checkpoint** — Gate a delegation workflow so tasks only go to trusted agents.
**Compliance Auditing** — Log trust scores before every external agent call.
<!-- END MANUAL -->

---
