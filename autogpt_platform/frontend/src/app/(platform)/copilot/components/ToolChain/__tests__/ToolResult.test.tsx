import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ChainRow } from "../helpers";
import { ToolResult } from "../ToolResult";

vi.mock("../../SetupRequirementsCard/SetupRequirementsCard", () => ({
  SetupRequirementsCard: ({
    credentialsLabel,
  }: {
    credentialsLabel?: string;
  }) => <div>{credentialsLabel}</div>,
}));

function row(output: unknown, tool?: string): ChainRow {
  return {
    key: "tool",
    category: "other",
    text: "Tool",
    state: "done",
    output,
    tool,
  };
}

describe("ToolResult", () => {
  afterEach(cleanup);

  it("renders a real unified diff", () => {
    render(<ToolResult row={row("@@ -1 +1 @@\n-old\n+new")} />);

    expect(screen.getByText("+1")).toBeDefined();
    expect(screen.getByText("-1")).toBeDefined();
    expect(screen.getByText("old")).toBeDefined();
    expect(screen.getByText("new")).toBeDefined();
  });

  it("renders JSON containing diff-like text as structured output", () => {
    render(
      <ToolResult
        row={row(JSON.stringify({ patch: "@@ -1 +1 @@\n-old\n+new" }))}
      />,
    );

    expect(screen.getByText("Patch")).toBeDefined();
    expect(screen.queryByText("+1")).toBeNull();
  });

  it("uses a fallback credentials label when an integration name is missing", () => {
    render(
      <ToolResult
        row={row(
          {
            message: "Connect an account",
            setup_info: { agent_id: "agent-1" },
          },
          "connect_integration",
        )}
      />,
    );

    expect(screen.getByText("Integration credentials")).toBeDefined();
    expect(screen.queryByText("undefined credentials")).toBeNull();
  });

  it.each([
    [
      "run_agent",
      { graph_name: "Research Agent", execution_id: "exec-1" },
      "Research Agent",
    ],
    ["find_block", { blocks: [{ name: "Web Search" }] }, "Web Search"],
    [
      "web_search",
      { results: [{ title: "Example result", url: "https://example.com" }] },
      "Example result",
    ],
    [
      "ask_question",
      { questions: [{ question: "Which region?", keyword: "region" }] },
      "Which region?",
    ],
    [
      "search_feature_requests",
      { results: [{ identifier: "FR-101", title: "Dark mode" }] },
      "Dark mode",
    ],
    [
      "create_agent",
      { agent_name: "Draft agent", node_count: 3 },
      "Draft agent",
    ],
    ["run_mcp_tool", { result: { identifier: "OPEN-3211" } }, "OPEN-3211"],
  ])("dispatches %s to its specialized card", (tool, output, expected) => {
    render(<ToolResult row={row(output, tool)} />);

    expect(screen.getByText(expected)).toBeDefined();
  });
});
