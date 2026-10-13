import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureMessage: vi.fn(),
  getTraceData: vi.fn(() => ({})),
}));

import * as Sentry from "@sentry/nextjs";
import { fireEvent } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  normalizeWhitespace,
} from "@/tests/integrations/test-utils";
import { RunAgentTool, type RunAgentToolPart } from "../RunAgent";
import { CopilotChatActionsProvider } from "../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";

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
  it("opens actual execution outputs inside the conversation", async () => {
    const graph = "10000000-0000-4000-8000-000000000001";
    const execution = "20000000-0000-4000-8000-000000000002";
    server.use(
      http.get(`*/api/graphs/${graph}/executions/${execution}`, () =>
        HttpResponse.json({
          id: execution,
          graph_id: graph,
          status: "COMPLETED",
          outputs: { report: ["Finished report"] },
        }),
      ),
    );
    render(
      <CopilotChatActionsProvider onSend={vi.fn()} chatSurface="copilot">
        <RunAgentTool
          part={makePart({
            state: "output-available",
            output: JSON.stringify({
              type: "execution_started",
              graph_id: graph,
              execution_id: execution,
              status: "QUEUED",
              message: "Started",
            }),
          })}
        />
      </CopilotChatActionsProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Status: QUEUED" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "View Execution" }),
    );
    expect(await screen.findByText("Finished report")).toBeDefined();
    expect(screen.getByText("Status: completed")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Hide Execution" }));
    expect(screen.queryByText("Finished report")).toBeNull();
  });
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
