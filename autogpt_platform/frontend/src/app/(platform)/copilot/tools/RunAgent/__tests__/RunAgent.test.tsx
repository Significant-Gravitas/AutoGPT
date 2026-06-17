import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureMessage: vi.fn(),
}));

import * as Sentry from "@sentry/nextjs";
import { render, screen } from "@/tests/integrations/test-utils";
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

// MorphingTextAnimation renders one span per character and uses a non-breaking
// space for spaces, so normalize all whitespace before asserting on the text.
function normalizedText(container: HTMLElement): string {
  return (container.textContent ?? "").replace(/\s/g, " ");
}

describe("RunAgentTool streaming state", () => {
  it("shows the plain loading line (no mini-game) while streaming", () => {
    const { container } = render(
      <RunAgentTool part={makePart({ state: "input-streaming" })} />,
    );

    expect(normalizedText(container)).toContain(
      "Running agent, this might take a minute",
    );
    expect(normalizedText(container)).not.toContain("Play while you wait");
    expect(normalizedText(container)).not.toContain("WASD");
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
