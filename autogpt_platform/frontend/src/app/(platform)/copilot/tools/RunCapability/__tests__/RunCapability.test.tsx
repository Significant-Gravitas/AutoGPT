import type { ToolUIPart } from "ai";
import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunCapabilityTool } from "../RunCapability";

// The three renderers this component routes between, stubbed down to the
// part they receive: what is under test is the routing and the re-shaping,
// not what a block or MCP card looks like.
vi.mock("../../GenericTool/GenericTool", () => ({
  GenericTool: () => <div data-testid="generic" />,
}));
vi.mock("../../RunBlock/RunBlock", () => ({
  RunBlockTool: ({ part }: { part: ToolUIPart }) => (
    <div data-testid="block">{JSON.stringify(part.input)}</div>
  ),
}));
vi.mock("../../RunMCPTool/RunMCPTool", () => ({
  RunMCPToolComponent: ({ part }: { part: ToolUIPart }) => (
    <div data-testid="mcp">{JSON.stringify(part.input)}</div>
  ),
}));

function part(input: unknown, output?: unknown): ToolUIPart {
  return {
    type: "tool-run_capability",
    toolCallId: "call-1",
    state: "output-available",
    input,
    output,
  } as ToolUIPart;
}

describe("RunCapabilityTool", () => {
  afterEach(cleanup);

  it("sends a block id to the block renderer, unprefixed", () => {
    render(
      <RunCapabilityTool
        part={part({ id: "block:abc-123", input: { url: "https://x" } })}
      />,
    );

    expect(JSON.parse(screen.getByTestId("block").textContent ?? "")).toEqual({
      block_id: "abc-123",
      input_data: { url: "https://x" },
      validate_only: false,
    });
  });

  it("sends an MCP call to the MCP renderer, flattened", () => {
    render(
      <RunCapabilityTool
        part={part(
          {
            id: "mcp:mcp.linear.app",
            input: { tool: "create_issue", arguments: { title: "t" } },
          },
          { type: "mcp_tool_output", server_url: "https://mcp.linear.app/mcp" },
        )}
      />,
    );

    expect(
      JSON.parse(screen.getByTestId("mcp").textContent ?? ""),
    ).toMatchObject({
      server_url: "https://mcp.linear.app/mcp",
      tool_name: "create_issue",
      tool_arguments: { title: "t" },
    });
  });

  it("renders a platform tool generically", () => {
    render(<RunCapabilityTool part={part({ id: "tool:list_schedules" })} />);

    expect(screen.getByTestId("generic")).toBeDefined();
  });

  it("renders a describe_capability answer generically, whatever the target", () => {
    render(
      <RunCapabilityTool
        part={part({ id: "block:abc-123" }, { type: "capability_details" })}
      />,
    );

    expect(screen.getByTestId("generic")).toBeDefined();
  });

  it("routes a resumed MCP call by its review id, before any output", () => {
    render(
      <RunCapabilityTool
        part={part({ review_id: "copilot-mcp-mcp.linear.app:ab12" })}
      />,
    );

    expect(screen.getByTestId("mcp")).toBeDefined();
  });

  it("falls back to the block renderer when nothing names the target", () => {
    render(<RunCapabilityTool part={part({})} />);

    expect(screen.getByTestId("block")).toBeDefined();
  });
});
