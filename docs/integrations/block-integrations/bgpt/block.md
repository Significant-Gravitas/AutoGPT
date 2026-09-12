# BGPT Scientific Evidence

<!-- MANUAL: file_description -->
Use BGPT through the MCP Tool block to retrieve structured scientific evidence — including study limitations, sample size, conflict-of-interest data, and falsifiability criteria — before summarizing a scientific claim.
<!-- END MANUAL -->

## BGPT via MCP Tool

### What it is

BGPT is a hosted MCP server that searches scientific papers and returns structured evidence fields (methods, sample size, results, limitations, conflict-of-interest statements, data/code availability, quality scores, and `how_to_falsify`) instead of just titles or abstracts.

### How it works

Connect to BGPT using the **MCP Tool** block (`run_mcp_tool` / `mcp/block.md`) with the hosted Streamable HTTP endpoint:

| Field | Value |
|---|---|
| Server URL | `https://bgpt.pro/mcp/stream` |
| Alt (SSE) | `https://bgpt.pro/mcp/sse` |
| Tool name | `search_papers` |

The free tier returns the first 50 results with no API key required.

### Example workflow

1. **MCP Tool** block — `server_url: https://bgpt.pro/mcp/stream`, `selected_tool: search_papers`, `tool_arguments: {"query": "GLP-1 alcohol craving"}`
2. The result separates review-level evidence from individual studies (e.g. a small human pilot), and includes each study's limitations, data availability, and `how_to_falsify` criteria.
3. Pass the result to an **AI** block to summarize the claim, explicitly citing study quality and limitations rather than treating all results as equally strong.

### Possible use case

<!-- MANUAL: use_case -->
- **Fact-checking scientific claims**: Before an agent states "X causes Y," query BGPT to confirm the claim is backed by more than a single small pilot study.
- **Research summarization with caveats**: Generate literature summaries that explicitly note sample size, conflicts of interest, and falsifiability instead of presenting abstracts as settled fact.
- **Evidence-graded content generation**: Build agents that grade the strength of scientific evidence before including it in articles, reports, or chat responses.
<!-- END MANUAL -->

### Links

- Docs: [https://bgpt.pro/mcp/](https://bgpt.pro/mcp/)
- GitHub: [https://github.com/connerlambden/bgpt-mcp](https://github.com/connerlambden/bgpt-mcp)
- OpenAPI spec: [bgpt-mcp openapi.yaml](https://raw.githubusercontent.com/connerlambden/bgpt-mcp/main/openapi.yaml)
- Evidence response shape: [EVIDENCE_DEMO.md](https://github.com/connerlambden/bgpt-mcp/blob/main/EVIDENCE_DEMO.md)

---
