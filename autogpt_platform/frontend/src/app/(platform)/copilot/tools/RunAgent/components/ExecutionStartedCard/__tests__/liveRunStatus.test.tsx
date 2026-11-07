import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { CopilotChatSurface } from "../../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { RunAgentTool } from "../../../RunAgent";
import { LIVE_STATUS_POLL_MS } from "../helpers";

const EXECUTION_URL = "/api/proxy/api/graphs/:graphId/executions/:graphExecId";

function makeRun(
  status: AgentExecutionStatus,
  error: string | null = null,
): GraphExecution {
  return {
    id: "exec-1",
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    status,
    stats: error ? { error } : null,
    outputs: {},
  };
}

function serveRuns(runs: GraphExecution[]) {
  const requests: string[] = [];
  server.use(
    http.get(EXECUTION_URL, ({ params }) => {
      requests.push(`${params.graphId}/${params.graphExecId}`);
      const run = runs[Math.min(requests.length - 1, runs.length - 1)];
      return HttpResponse.json(run);
    }),
  );
  return requests;
}

function renderRun(
  status: string,
  chatSurface: CopilotChatSurface = "copilot",
) {
  return render(
    <CopilotChatActionsProvider onSend={vi.fn()} chatSurface={chatSurface}>
      <RunAgentTool
        part={{
          type: "tool-run_agent",
          toolCallId: "call-1",
          state: "output-available",
          input: { username_agent_slug: "user/flux-image" },
          output: {
            type: "execution_started",
            message: "Workflow 'Flux Image' execution started successfully.",
            execution_id: "exec-1",
            graph_id: "graph-1",
            graph_name: "Flux Image",
            library_agent_link: "/library/agents/lib-1",
            status,
          },
        }}
      />
    </CopilotChatActionsProvider>,
  );
}

function expandCard() {
  fireEvent.click(screen.getByRole("button", { name: /Flux Image/ }));
}

describe("run card started from chat", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows a run that fails after the turn as failed, with its error", async () => {
    const requests = serveRuns([
      makeRun("RUNNING"),
      makeRun("FAILED", "Missing API key for the image provider"),
    ]);
    renderRun("QUEUED");

    expect(await screen.findByText("Status: RUNNING")).toBeDefined();

    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS));

    expect(await screen.findByText("Status: FAILED")).toBeDefined();
    expandCard();
    expect(screen.getByText("Execution failed")).toBeDefined();
    expect(screen.queryByText(/started successfully/)).toBeNull();
    expect(
      screen.getByText("Missing API key for the image provider"),
    ).toBeDefined();
    expect(requests).toEqual(["graph-1/exec-1", "graph-1/exec-1"]);

    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 3));
    expect(requests).toHaveLength(2);
  });

  it("shows a run that succeeds after the turn as completed", async () => {
    serveRuns([makeRun("COMPLETED")]);
    renderRun("QUEUED");

    expect(await screen.findByText("Status: COMPLETED")).toBeDefined();
    expandCard();
    expect(screen.getByText("Execution completed")).toBeDefined();
  });

  it.each([
    ["a run that already ended", "COMPLETED", "copilot"],
    ["a scheduled run", "SCHEDULED", "copilot"],
    ["the public share page", "QUEUED", "share"],
    ["the builder panel", "QUEUED", "builder"],
  ] as const)("makes no request for %s", async (_, status, surface) => {
    const requests = serveRuns([makeRun("FAILED", "should not show")]);
    renderRun(status, surface);

    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 2));

    expect(requests).toHaveLength(0);
    expect(screen.getByText(`Status: ${status}`)).toBeDefined();
  });

  it("keeps the started status when the run can't be read", async () => {
    let calls = 0;
    server.use(
      http.get(EXECUTION_URL, () => {
        calls += 1;
        return HttpResponse.json({ detail: "Not found" }, { status: 404 });
      }),
    );
    renderRun("QUEUED");

    await waitFor(() => expect(calls).toBe(1));
    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 2));

    expect(calls).toBe(1);
    expect(screen.getByText("Status: QUEUED")).toBeDefined();
  });

  it("stops polling when a later request gets a 404", async () => {
    let calls = 0;
    server.use(
      http.get(EXECUTION_URL, () => {
        calls += 1;
        if (calls === 1) return HttpResponse.json(makeRun("RUNNING"));
        return HttpResponse.json({ detail: "Not found" }, { status: 404 });
      }),
    );
    renderRun("QUEUED");

    expect(await screen.findByText("Status: RUNNING")).toBeDefined();
    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS));
    await waitFor(() => expect(calls).toBe(2));
    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 3));

    expect(calls).toBe(2);
  });

  it("keeps polling through a server error", async () => {
    let calls = 0;
    server.use(
      http.get(EXECUTION_URL, () => {
        calls += 1;
        if (calls === 1)
          return HttpResponse.json({ detail: "Bad gateway" }, { status: 502 });
        return HttpResponse.json(makeRun("FAILED", "Out of credits"));
      }),
    );
    renderRun("QUEUED");

    await waitFor(() => expect(calls).toBe(1));
    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS));

    expect(await screen.findByText("Status: FAILED")).toBeDefined();
  });
});
