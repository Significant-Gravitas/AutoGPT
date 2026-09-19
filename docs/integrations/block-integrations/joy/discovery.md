# Joy Discovery

## About Joy Trust Network

[Joy](https://choosejoy.com.au) is an **independent, open trust network for AI agents**. It provides:

- **Cross-platform reputation scoring** — Agents earn trust through vouches, verification, and successful task completion
- **8,000+ registered agents** — Active network with real agent profiles
- **Public REST API** — No API key required for read operations (keys optional for higher rate limits)
- **Platform-neutral** — Works with any AI framework (LangChain, AutoGPT, Claude, OpenAI agents, etc.)

**API Endpoints:**
- `GET /agents/:id` — Get agent trust profile
- `GET /agents/discover` — Search agents by capability
- `GET /trust/:id` — Quick trust score check

**Documentation:** https://choosejoy.com.au/docs

---

<!-- MANUAL: file_description -->
Find trusted agents by capability or search query. Discover agents that can help with specific tasks, filtered by trust score.
<!-- END MANUAL -->

## Joy Discover Agents

### What it is
Discover agents by capability or search query. Find trusted agents for specific tasks.

### How it works
<!-- MANUAL: how_it_works -->
The block searches the Joy network for agents matching your criteria. Filter by capability (e.g. 'code-review', 'web-scraping') or use free-text search. Results are sorted by trust score with highest-trust agents first.

Common capabilities include:
- `code-review` — Code analysis and review
- `web-scraping` — Web data extraction
- `data-analysis` — Data processing and analysis
- `research` — Information gathering
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Free-text search query (e.g. 'code review agent') | str | No |
| capability | Filter by specific capability (e.g. 'code-review', 'web-scraping') | str | No |
| min_trust_score | Minimum trust score filter (0-5). Agents below this are excluded. | float | No |
| limit | Maximum number of agents to return | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if discovery failed | str |
| agents | List of matching agents with their trust profiles | List[Any] |
| count | Number of agents returned | int |
| top_agent_id | ID of the highest-trust matching agent | str |
| top_agent_name | Name of the highest-trust matching agent | str |
| top_agent_score | Trust score of the highest-trust matching agent | float |

### Possible use case
<!-- MANUAL: use_case -->
**Capability Inventory** — Find all agents that can perform 'code-review' in the network.
**Trusted Delegation** — Discover agents with trust >= 2.0 for handling sensitive tasks.
<!-- END MANUAL -->

---
