import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureMessage: vi.fn(),
}));

import * as Sentry from "@sentry/nextjs";
import type { ToolUIPart } from "ai";
import { render, screen } from "@/tests/integrations/test-utils";
import { ConnectIntegrationTool } from "../ConnectIntegrationTool";

function makePart(overrides: Record<string, unknown> = {}): ToolUIPart {
  return {
    type: "tool-connect_integration",
    toolCallId: "call-connect-1",
    state: "input-streaming",
    input: { provider: "github" },
    ...overrides,
  } as ToolUIPart;
}

describe("ConnectIntegrationTool", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("styles the label as an error and reports when output is corrupted", () => {
    const { container } = render(
      <ConnectIntegrationTool
        part={makePart({
          state: "output-available",
          output: '{"setup_info":{"agent_name":"GitHub"',
        })}
      />,
    );
    expect(
      screen.getByText(/sign-in card data arrived corrupted/i),
    ).not.toBeNull();
    // isError now includes isCorrupted, so the label switches to the error
    // copy. MorphingTextAnimation splits chars and uses NBSP for spaces.
    const normalized = (container.textContent ?? "").replace(/ /g, " ");
    expect(normalized).toContain("Failed to connect");
    expect(vi.mocked(Sentry.captureMessage)).toHaveBeenCalledTimes(1);
  });

  it("shows the failure message on a genuine error without corrupted telemetry", () => {
    render(
      <ConnectIntegrationTool
        part={makePart({
          state: "output-error",
          output: '{"message":"OAuth denied"}',
        })}
      />,
    );
    expect(screen.getByText("OAuth denied")).not.toBeNull();
    expect(
      screen.queryByText(/sign-in card data arrived corrupted/i),
    ).toBeNull();
    expect(vi.mocked(Sentry.captureMessage)).not.toHaveBeenCalled();
  });
});
