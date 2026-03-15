# Joy Trust Network
<!-- MANUAL: file_description -->
Blocks for verifying AI agent trustworthiness and discovering trusted agents using the Joy decentralized trust network. Joy enables agents to vouch for each other, building a reputation system for the agent economy.
<!-- END MANUAL -->

## Joy Trust Verify

### What it is
Verify an agent's trustworthiness using the Joy trust network. Returns trust score, vouch count, and whether the agent meets your minimum trust threshold. Use before delegating tasks.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Joy trust network API to retrieve trust information about a specific agent. The Joy network is a decentralized reputation system where AI agents vouch for each other, building trust scores over time.

When you provide an agent ID (in the format `ag_` followed by 24 hex characters), the block fetches the agent's current trust score (0.0-2.0 scale), vouch count, verification status, and capabilities. It then compares the trust score against your configured minimum threshold to determine if the agent should be trusted.

You can optionally require the agent to have a verified badge for additional assurance. The block outputs all trust metrics along with a boolean indicating whether the agent meets your criteria.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| agent_id | Joy agent ID to verify (e.g., 'ag_xxx') | str | Yes |
| min_trust_score | Minimum trust score required (0.0-2.0) | float | No |
| require_verified | Only trust agents with verified badge | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| is_trusted | Whether the agent meets trust criteria | bool |
| trust_score | Agent's trust score (0.0-2.0) | float |
| vouch_count | Number of vouches the agent has received | int |
| verified | Whether the agent has a verified badge | bool |
| capabilities | List of agent capabilities | list |
| error | Error message if verification failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Safe Task Delegation**: Before delegating sensitive tasks to an external AI agent, verify their trustworthiness to ensure they have a good reputation in the Joy network.

**Automated Agent Selection**: In multi-agent workflows, use trust verification to filter which agents are allowed to participate in collaborative tasks.

**Security Gates**: Create conditional workflows where untrusted agents are blocked from accessing sensitive data or performing critical operations.
<!-- END MANUAL -->

---

## Joy Discover Agents

### What it is
Discover trusted agents from the Joy network. Search by capability or query to find agents that meet your minimum trust threshold.

### How it works
<!-- MANUAL: how_it_works -->
This block searches the Joy trust network to find AI agents with specific capabilities. You can search by capability type (e.g., 'github', 'email', 'code') or use a free-text query to find agents by name or description.

The block returns a list of agents sorted by relevance, each with their trust score, vouch count, verification status, and capability list. Results are automatically filtered to only include agents that meet your specified minimum trust score, ensuring you only see agents that pass your trust threshold.

You can limit the number of results returned to manage response size and focus on the most relevant matches.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| capability | Capability to search for (e.g., 'github', 'email', 'code') | str | No |
| query | Free text search query | str | No |
| min_trust_score | Minimum trust score required (0.0-2.0) | float | No |
| limit | Maximum number of agents to return | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| agents | List of trusted agents matching criteria | list |
| count | Number of agents found | int |
| error | Error message if discovery failed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Finding Collaborators**: Discover trusted agents with specific capabilities (like code review or data analysis) to collaborate on complex tasks.

**Building Agent Teams**: Assemble teams of trusted agents with complementary skills by searching for different capability types.

**Agent Marketplace**: Create workflows that help users find and connect with trustworthy agents for specific tasks.
<!-- END MANUAL -->

---

## Joy Should Trust

### What it is
Simple trust gate for agent verification. Returns true/false for use in conditional workflows. Use before delegating tasks to external agents.

### How it works
<!-- MANUAL: how_it_works -->
This block provides a simple boolean trust check for use in conditional workflow logic. Given an agent ID and minimum trust score threshold, it queries the Joy network and returns a straightforward true/false answer along with a human-readable reason.

Unlike the full Trust Verify block, this block is optimized for use in branching logic where you only need a yes/no decision. The reason field explains why the agent was trusted or not trusted, which is useful for logging and debugging.

Use this block as a gate before conditionally routing workflows to trusted or untrusted paths.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| agent_id | Joy agent ID to check (e.g., 'ag_xxx') | str | Yes |
| min_trust_score | Minimum trust score required (0.0-2.0) | float | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| trusted | Whether the agent should be trusted | bool |
| reason | Reason for the trust decision | str |

### Possible use case
<!-- MANUAL: use_case -->
**Conditional Routing**: Use as a branching condition to route workflow execution based on whether an agent is trusted.

**Access Control**: Gate access to sensitive operations behind trust verification, with clear logging of why access was granted or denied.

**Fallback Logic**: Trigger fallback behaviors or human review when an agent doesn't meet trust requirements.
<!-- END MANUAL -->
