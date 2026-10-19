import fs from "fs";
import path from "path";
import { describe, expect, it } from "vitest";
import { buildGraph, buildRunningExecution } from "../lib/builder-fixture";
import { replyText } from "../lib/copilot-fixture";
import {
  buildLibraryAgents,
  buildRunningExecutions,
} from "../lib/library-fixture";
import { FIXTURES_DIR } from "../lib/paths";
import { splitFrames } from "../lib/stream-server";
import {
  REPLY_20_LINES,
  REPLY_LONG_CODE,
  buildReplyStream,
  buildSessionRows,
} from "../fixtures/generate";

describe("copilot fixtures", () => {
  it("builds a 200-row session with text, code, tables and tool calls", () => {
    const rows = buildSessionRows();
    expect(rows).toHaveLength(200);
    expect(rows.map((row) => row.sequence)).toEqual(rows.map((_, i) => i));
    expect(rows.some((row) => row.role === "tool")).toBe(true);
    expect(rows.some((row) => row.content.includes("```ts"))).toBe(true);
    expect(rows.some((row) => row.content.includes("| Day |"))).toBe(true);
  });

  it("streams the 20-line reply as one text part that ends with [DONE]", () => {
    expect(REPLY_20_LINES.split("\n")).toHaveLength(20);
    const sse = buildReplyStream(REPLY_20_LINES, "m1");
    const frames = splitFrames(sse);
    expect(frames[0]).toBe('data: {"type":"start","messageId":"m1"}');
    expect(frames.at(-1)).toBe("data: [DONE]");
    expect(replyText(sse)).toBe(REPLY_20_LINES);
  });

  it("reads the reply text out of recorded frames with id lines and comments", () => {
    const recorded = [
      'id: t:1-0\ndata: {"type": "data-status", "data": {}}',
      ": heartbeat",
      'id: t:2-0\ndata: {"type":"text-delta","id":"x","delta":"Hello "}',
      'id: t:2-1\ndata: {"type":"text-delta","id":"x","delta":"world"}',
      "data: [DONE]",
    ].join("\n\n");
    expect(replyText(recorded)).toBe("Hello world");
  });

  it("matches the committed fixture files", () => {
    const session = JSON.parse(
      fs.readFileSync(path.join(FIXTURES_DIR, "session-200.json"), "utf8"),
    );
    expect(session.messages).toEqual(buildSessionRows());
    expect(
      fs.readFileSync(path.join(FIXTURES_DIR, "reply-20-lines.sse"), "utf8"),
    ).toBe(buildReplyStream(REPLY_20_LINES, "perf-reply-20-lines"));
    expect(
      fs.readFileSync(path.join(FIXTURES_DIR, "reply-long-code.sse"), "utf8"),
    ).toBe(buildReplyStream(REPLY_LONG_CODE, "perf-reply-long-code"));
  });
});

describe("builder fixture", () => {
  it("owns a 50-node graph by the signed-in user with a running execution", () => {
    const graph = buildGraph("user-1");
    const execution = buildRunningExecution(graph, "user-1");
    expect(graph.user_id).toBe("user-1");
    expect(graph.nodes).toHaveLength(50);
    const ids = new Set(graph.nodes.map((node) => node.id));
    expect(ids.size).toBe(50);
    expect(
      graph.links.every(
        (link) => ids.has(link.source_id) && ids.has(link.sink_id),
      ),
    ).toBe(true);
    expect(execution.status).toBe("RUNNING");
    expect(
      execution.node_executions.filter((node) => node.status === "RUNNING"),
    ).toHaveLength(3);
  });
});

describe("library fixture", () => {
  it("gives each of the 20 workflows a RUNNING execution", () => {
    const agents = buildLibraryAgents();
    const executions = buildRunningExecutions(agents, "user-1");
    expect(agents).toHaveLength(20);
    expect(executions.map((execution) => execution.graph_id)).toEqual(
      agents.map((agent) => agent.graph_id),
    );
    expect(
      executions.every((execution) => execution.status === "RUNNING"),
    ).toBe(true);
  });
});
