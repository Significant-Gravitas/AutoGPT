"""Dream-pass evaluation harness (P0.6).

Snapshots cost / latency / quality metrics against a stable input so we
can tell whether changes to the dream pipeline (P0.3a stale-fact, P0.4
ratification, P0.5 web fact-check) are net positive or net regressions.

Architecture: AgentProbe (TS, separate repo at ~/code/agpt/AgentProbe)
owns the scorers and the runner for retrieval / demotion / dedup /
procedure. This package provides the Python orchestration shim
(``dream-eval`` CLI) plus our own metrics for ratification / cost /
latency that are tightly coupled to the dream-orchestrator shapes.

The shim shells out to ``bun run agentprobe -- ...``, captures the JSON
output, decorates with platform metadata (graphiti_version, git_sha,
transport), and writes ``eval/results.json``.

See ``dream/p0-spec.md`` §7 for the spec.
"""
