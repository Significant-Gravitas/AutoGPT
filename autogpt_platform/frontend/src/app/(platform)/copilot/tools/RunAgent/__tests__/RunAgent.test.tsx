import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureMessage: vi.fn(),
}));

import * as Sentry from "@sentry/nextjs";
import {
  render,
  screen,
  normalizeWhitespace,
} from "@/tests/integrations/test-utils";
import { RunAgentTool, type RunAgentToolPart } from "../RunAgent";

function makePart(overrides: Partial<RunAgentToolPart> = {}): RunAgentToolPart {
  return {
    type: "tool-run_agent",
    toolCallId: "call-agent-1",
    state: "input-streaming",
    input: { agent_id: "a1" },
    ...overrides,
  };
}

describe("RunAgentTool streaming state", () => {
  it("shows the plain loading line (no mini-game) while streaming", () => {
    const { container } = render(
      <RunAgentTool part={makePart({ state: "input-streaming" })} />,
    );

    expect(normalizeWhitespace(container)).toContain(
      "Running agent, this might take a minute",
    );
    expect(normalizeWhitespace(container)).not.toContain("Play while you wait");
    expect(normalizeWhitespace(container)).not.toContain("WASD");
  });

  it("shows the scheduling loading line when the input is a schedule", () => {
    const { container } = render(
      <RunAgentTool
        part={makePart({
          state: "input-streaming",
          input: { agent_id: "a1", cron: "0 0 * * *" },
        })}
      />,
    );

    expect(normalizeWhitespace(container)).toContain(
      "Scheduling agent, this might take a minute",
    );
    expect(normalizeWhitespace(container)).not.toContain("Running agent");
  });
});

describe("RunAgentTool corrupted output", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a visible error and reports to Sentry when output is unparseable JSON", () => {
    render(
      <RunAgentTool
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
      <RunAgentTool
        part={makePart({
          state: "output-available",
          output: JSON.stringify({
            type: "execution_started",
            execution_id: "e1",
            message: "Started",
          }),
        })}
      />,
    );
    expect(screen.queryByText(/result data arrived corrupted/i)).toBeNull();
    expect(vi.mocked(Sentry.captureMessage)).not.toHaveBeenCalled();
  });
});
