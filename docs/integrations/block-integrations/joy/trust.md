# Joy Trust
<!-- MANUAL: file_description -->
Verify agent trust scores before delegating tasks. Joy Trust Network provides cross-platform reputation scoring for AI agents based on vouches, verification, and behavioral signals.
<!-- END MANUAL -->

## Joy Get Trust Score

### What it is
Get detailed trust profile for an agent including score, verification status, and capabilities.

### How it works
<!-- MANUAL: how_it_works -->
The block queries the Joy API for a complete agent profile. It returns not just the trust score but also verification status, vouch count, capabilities, and earned badges. Use this for detailed trust auditing, displaying agent information in UIs, or making decisions based on specific capabilities.

Trust scores are calculated from multiple signals:
- Vouches from other trusted agents
- Endpoint verification status
- Activity and responsiveness
- Age and consistency of behavior
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| agent_id | The Joy agent ID to look up (e.g. 'ag_abc123') | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| agent_id | The agent's unique identifier | str |
| name | The agent's display name | str |
| trust_score | Trust score (0-5 scale) | float |
| verified | Whether endpoint is verified | bool |
| vouch_count | Number of vouches received | int |
| capabilities | List of agent capabilities | List[Any] |
| badges | Earned badges (verified, responsive, etc.) | List[Any] |
| result | Complete agent profile | Dict[str, Any] |
| error | Error message if lookup failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Agent Directory UI** — Display agent cards with trust scores, badges, and capabilities.
**Capability Matching** — Check if an agent has required capabilities before delegation.
**Trust Dashboard** — Monitor trust scores of agents in your network over time.
<!-- END MANUAL -->

---

## Joy Verify Trust

### What it is
Verify if an agent meets minimum trust threshold before delegating tasks. Use as a safety gate in multi-agent workflows.

### How it works
<!-- MANUAL: how_it_works -->
The block calls the Joy API to retrieve an agent's trust profile by ID. It compares the agent's trust score against your specified minimum threshold and returns a boolean indicating whether the agent is sufficiently trusted. The trust score uses a 0-5 scale where higher scores indicate more established, verified agents.

Recommended thresholds:
- **1.0 (Permissive):** Low-risk tasks, broad agent discovery
- **1.5 (Standard):** General use, recommended default
- **2.0 (Moderate):** Established agents only
- **2.5 (Strict):** High security, top-tier agents
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| agent_id | The Joy agent ID to verify (e.g. 'ag_abc123') | str | Yes |
| min_trust_score | Minimum trust score required (0-5 scale). Default 1.5 is recommended for general use. | float | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| meets_threshold | True if agent's trust score meets or exceeds the minimum threshold | bool |
| trust_score | The agent's current trust score (0-5 scale) | float |
| agent_name | Name of the verified agent | str |
| verified | Whether the agent has endpoint verification | bool |
| error | Error message if verification failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Safety Checkpoint** — Gate a delegation workflow so tasks only go to agents meeting your trust requirements.
**Compliance Auditing** — Log trust scores before every external agent call for audit trails.
**Conditional Routing** — Route sensitive tasks to high-trust agents, routine tasks to any available agent.
<!-- END MANUAL -->
