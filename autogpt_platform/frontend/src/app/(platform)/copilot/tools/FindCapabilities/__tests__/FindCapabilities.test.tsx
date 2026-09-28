import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup, fireEvent } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { FindCapabilitiesTool } from "../FindCapabilities";
import type { CapabilityListOutput, FindCapabilityToolPart } from "../helpers";

const OUTPUT: CapabilityListOutput = {
  type: "capability_list",
  query: "linear issue",
  count: 2,
  capabilities: [
    {
      id: "mcp:mcp.linear.app",
      name: "Linear",
      purpose: "Issues, projects and cycles",
      kind: "mcp_server",
      connected: true,
    },
    {
      id: "block:abc-123",
      name: "Send Web Request",
      purpose: "Call an HTTP endpoint",
      kind: "block",
      connected: false,
    },
  ],
  fallback: [
    {
      id: "block:def-456",
      name: "HTTP Request",
      purpose: "A lower-level request",
      kind: "block",
    },
  ],
};

function part(overrides: Partial<FindCapabilityToolPart>) {
  return {
    type: "tool-find_capability",
    toolCallId: "call-1",
    state: "output-available",
    input: { query: "linear issue" },
    ...overrides,
  } as FindCapabilityToolPart;
}

describe("FindCapabilitiesTool", () => {
  afterEach(cleanup);

  it("shows results and fallbacks together, counting only the hits", () => {
    render(<FindCapabilitiesTool part={part({ output: OUTPUT })} />);
    fireEvent.click(screen.getByText("Results"));

    expect(screen.getByText("Linear")).toBeDefined();
    expect(screen.getByText("Send Web Request")).toBeDefined();
    // The fallback rides along in the same row — it is still something the
    // model can call, just not what the query asked for.
    expect(screen.getByText("HTTP Request")).toBeDefined();
    expect(screen.getByText('2 results for "linear issue"')).toBeDefined();
  });

  it("marks connection state only where the capability needs one", () => {
    render(<FindCapabilitiesTool part={part({ output: OUTPUT })} />);
    fireEvent.click(screen.getByText("Results"));

    expect(screen.getByText("connected")).toBeDefined();
    expect(screen.getByText("sign in")).toBeDefined();
    // The fallback reports no connection field, so it gets no badge: three
    // cards, two badges.
    expect(screen.queryAllByText(/^(connected|sign in)$/)).toHaveLength(2);
  });

  it("labels a skill as one, not as an action", () => {
    const withSkill = {
      ...OUTPUT,
      count: 1,
      capabilities: [
        {
          id: "skill:triage-and-prioritize",
          name: "triage-and-prioritize",
          purpose: "Triage a support ticket",
          kind: "skill",
        },
      ],
      fallback: [],
    };
    render(<FindCapabilitiesTool part={part({ output: withSkill })} />);
    fireEvent.click(screen.getByText("Results"));

    expect(screen.getByText("triage-and-prioritize")).toBeDefined();
    expect(screen.getByText("skill")).toBeDefined();
    expect(screen.queryByText("action")).toBeNull();
  });

  it("singularises a lone result", () => {
    const single = {
      ...OUTPUT,
      count: 1,
      capabilities: [OUTPUT.capabilities[0]],
    };
    render(<FindCapabilitiesTool part={part({ output: single })} />);

    expect(screen.getByText('1 result for "linear issue"')).toBeDefined();
  });

  it("announces the search while it streams and shows no results row", () => {
    render(<FindCapabilitiesTool part={part({ state: "input-streaming" })} />);

    expect(
      screen.getByText('Searching capabilities for "linear issue"'),
    ).toBeDefined();
    expect(screen.queryByText("Results")).toBeNull();
  });

  it("reports a failed search without pretending it returned nothing", () => {
    render(<FindCapabilitiesTool part={part({ state: "output-error" })} />);

    expect(screen.getByText('Search failed for "linear issue"')).toBeDefined();
  });

  it("survives an output that is not a capability list", () => {
    render(
      <FindCapabilitiesTool part={part({ output: { type: "block_list" } })} />,
    );

    expect(screen.queryByText("Results")).toBeNull();
  });
});
