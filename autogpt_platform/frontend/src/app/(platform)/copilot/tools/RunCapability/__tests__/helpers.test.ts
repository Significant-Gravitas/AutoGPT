import type { ToolUIPart } from "ai";
import { describe, expect, it } from "vitest";
import {
  asBlockPart,
  asMcpPart,
  capabilityId,
  isCapabilityDetails,
  isMcpCapability,
  parseOutputObject,
} from "../helpers";

function part(input: unknown, output?: unknown): ToolUIPart {
  return {
    type: "tool-run_capability",
    toolCallId: "call-1",
    state: "output-available",
    input,
    output,
  } as ToolUIPart;
}

describe("run_capability helpers", () => {
  it("parses string and object outputs", () => {
    expect(parseOutputObject('{"type":"block_output"}')).toEqual({
      type: "block_output",
    });
    expect(parseOutputObject({ type: "x" })).toEqual({ type: "x" });
    expect(parseOutputObject("not json")).toBeNull();
    expect(parseOutputObject(undefined)).toBeNull();
  });

  it("recognises MCP targets from the id or the response", () => {
    expect(isMcpCapability("mcp:mcp.linear.app", null)).toBe(true);
    expect(isMcpCapability("https://mcp.example.com/mcp", null)).toBe(true);
    expect(isMcpCapability("block:abc", { type: "mcp_tool_output" })).toBe(
      true,
    );
    expect(isMcpCapability("block:abc", { type: "block_output" })).toBe(false);
    expect(isCapabilityDetails({ type: "capability_details" })).toBe(true);
  });

  it("re-shapes a block call into the run_block part shape", () => {
    const shaped = asBlockPart(
      part({ id: "block:abc-123", input: { url: "https://x" } }),
    );
    expect(shaped.type).toBe("tool-run_block");
    expect(shaped.input).toEqual({
      block_id: "abc-123",
      input_data: { url: "https://x" },
      validate_only: false,
    });
    expect(capabilityId({ id: " block:abc " })).toBe("block:abc");
  });

  it("re-shapes an MCP call into the run_mcp_tool part shape", () => {
    const shaped = asMcpPart(
      part(
        {
          id: "mcp:mcp.linear.app",
          input: { tool: "create_issue", arguments: { title: "t" } },
        },
        { type: "mcp_tool_output", server_url: "https://mcp.linear.app/mcp" },
      ),
      { type: "mcp_tool_output", server_url: "https://mcp.linear.app/mcp" },
    );
    expect(shaped.type).toBe("tool-run_mcp_tool");
    expect(shaped.input).toEqual({
      server_url: "https://mcp.linear.app/mcp",
      tool_name: "create_issue",
      tool_arguments: { title: "t" },
      surface_connect_card: false,
    });
  });

  it("reads a resumed call's target out of its review id", () => {
    expect(capabilityId({ review_id: "copilot-mcp-mcp.linear.app:ab12" })).toBe(
      "mcp:mcp.linear.app",
    );
    expect(capabilityId({ review_id: "copilot-node-abc-123:ab12" })).toBe(
      "block:abc-123",
    );
    expect(capabilityId({ review_id: "something-else:ab12" })).toBe("");
    expect(capabilityId({ review_id: "copilot-mcp-no-suffix" })).toBe("");
  });

  it("routes a resumed MCP call to the MCP renderer before its output lands", () => {
    const id = capabilityId({ review_id: "copilot-mcp-mcp.linear.app:ab12" });
    expect(isMcpCapability(id, null)).toBe(true);
  });

  it("uses a bare server URL id when the response has none", () => {
    const shaped = asMcpPart(
      part({ id: "https://mcp.example.com/mcp", input: { connect: true } }),
      null,
    );
    expect(shaped.input).toMatchObject({
      server_url: "https://mcp.example.com/mcp",
      surface_connect_card: true,
    });
  });
});
