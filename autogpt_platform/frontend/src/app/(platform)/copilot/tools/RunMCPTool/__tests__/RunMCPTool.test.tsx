import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureMessage: vi.fn(),
}));

import * as Sentry from "@sentry/nextjs";
import { render, screen } from "@/tests/integrations/test-utils";
import { RunMCPToolComponent, type RunMCPToolPart } from "../RunMCPTool";

function makePart(overrides: Partial<RunMCPToolPart> = {}): RunMCPToolPart {
  return {
    type: "tool-run_mcp_tool",
    toolCallId: "call-mcp-1",
    state: "input-streaming",
    input: { server_url: "https://mcp.example.com" },
    ...overrides,
  };
}

describe("RunMCPToolComponent corrupted output", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a visible error and reports to Sentry when output is unparseable JSON", () => {
    render(
      <RunMCPToolComponent
        part={makePart({
          state: "output-available",
          output: '{"type":"setup_requirements","message":"Connect Goo',
        })}
      />,
    );
    expect(screen.getByText(/result data arrived corrupted/i)).not.toBeNull();
    expect(vi.mocked(Sentry.captureMessage)).toHaveBeenCalledTimes(1);
  });

  it("does not flag a valid output as corrupted", () => {
    render(
      <RunMCPToolComponent
        part={makePart({
          state: "output-available",
          output: JSON.stringify({
            tool_name: "search",
            server_url: "https://mcp.example.com",
            result: "ok",
          }),
        })}
      />,
    );
    expect(screen.queryByText(/result data arrived corrupted/i)).toBeNull();
    expect(vi.mocked(Sentry.captureMessage)).not.toHaveBeenCalled();
  });
});
