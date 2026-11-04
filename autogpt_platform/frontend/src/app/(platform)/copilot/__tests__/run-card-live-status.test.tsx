import { http, HttpResponse } from "msw";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getGetV2GetPendingReviewsForChatSessionMockHandler,
  getGetV2GetPendingReviewsForExecutionMockHandler,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { BuilderChatPanel } from "@/app/(platform)/build/components/BuilderChatPanel/BuilderChatPanel";
import { useBuilderChatPanel } from "@/app/(platform)/build/components/BuilderChatPanel/useBuilderChatPanel";
import { ChatContainer } from "../components/ChatContainer/ChatContainer";
import { LIVE_STATUS_POLL_MS } from "../tools/RunAgent/components/ExecutionStartedCard/helpers";

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => <textarea aria-label="Chat input" />,
}));

vi.mock(
  "@/app/(platform)/copilot/components/UsageLimits/useIsUsageLimitReached",
  () => ({ useIsUsageLimitReached: () => false }),
);

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => ({
  ...(await importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
}));

// The chat's pending-review poller reads the newest run through the same
// query. It has tests of its own; left out here so every request counted
// below comes from the run card.
vi.mock(
  "@/app/(platform)/copilot/components/CopilotPendingReviews/CopilotPendingReviews",
  () => ({ CopilotPendingReviews: () => null }),
);

vi.mock(
  "@/app/(platform)/build/components/BuilderChatPanel/useBuilderChatPanel",
  () => ({ useBuilderChatPanel: vi.fn() }),
);

type Messages = UIMessage<unknown, UIDataTypes, UITools>[];

const EXECUTION_URL = "/api/proxy/api/graphs/:graphId/executions/:graphExecId";
const RUN_ERROR = "Missing API key for the image provider";

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
      return HttpResponse.json(
        runs[Math.min(requests.length - 1, runs.length - 1)],
      );
    }),
  );
  return requests;
}

/** The turn as the chat stores it: Otto started the run and didn't wait. */
function transcript(
  status: string,
  via: "run_agent" | "run_capability" = "run_agent",
): Messages {
  const agentInput = { username_agent_slug: "user/flux-image" };
  return [
    {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Make me a random image" }],
    },
    {
      id: "assistant-1",
      role: "assistant",
      parts: [
        {
          type: `tool-${via}`,
          toolCallId: "call-1",
          state: "output-available",
          input:
            via === "run_agent"
              ? agentInput
              : { id: "tool:run_agent", input: agentInput },
          output: {
            type: "execution_started",
            message: "Workflow 'Flux Image' execution started successfully.",
            execution_id: "exec-1",
            graph_id: "graph-1",
            graph_name: "Flux Image",
            library_agent_link: "/library/agents/lib-1",
            status,
          },
        },
      ],
    } as unknown as Messages[number],
  ];
}

function renderMainChat(messages: Messages) {
  return render(
    <ChatContainer
      messages={messages}
      status="ready"
      error={undefined}
      sessionId="session-1"
      isLoadingSession={false}
      isCreatingSession={false}
      onCreateSession={vi.fn()}
      onSend={vi.fn()}
      onStop={vi.fn()}
    />,
  );
}

function renderBuilderChat(messages: Messages) {
  vi.mocked(useBuilderChatPanel).mockReturnValue({
    isOpen: true,
    handleToggle: vi.fn(),
    sessionId: "session-1",
    messages,
    status: "ready",
    error: undefined,
    stop: vi.fn(),
    onSend: vi.fn(),
    queuedMessages: [],
    isTurnSettling: false,
    isBootstrapping: false,
    revertTargetVersion: null,
    handleRevert: vi.fn(),
    bindError: null,
    bootstrapError: null,
    retryBind: vi.fn(),
    retryBootstrap: vi.fn(),
  } as unknown as ReturnType<typeof useBuilderChatPanel>);
  return render(<BuilderChatPanel />);
}

/** Opens the finished chain, then the run's row, the way a user reaches the
 *  run card after the turn. Returns the card. */
function openRunCard() {
  const thread = screen.getByRole("log");
  const [chainToggle] = within(thread).getAllByRole("button", {
    expanded: false,
  });
  fireEvent.click(chainToggle);
  fireEvent.click(within(thread).getByRole("button", { expanded: false }));
  return within(thread).getByTestId("execution-card");
}

class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.stubGlobal("ResizeObserver", MockResizeObserver);
  server.use(
    getGetV2GetPendingReviewsForChatSessionMockHandler([]),
    getGetV2GetPendingReviewsForExecutionMockHandler([]),
  );
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("run card in the chat thread", () => {
  it("turns a run that fails after the turn to failed, with its error", async () => {
    const requests = serveRuns([
      makeRun("RUNNING"),
      makeRun("FAILED", RUN_ERROR),
    ]);
    renderMainChat(transcript("QUEUED"));
    const card = openRunCard();

    expect(await within(card).findByText("running")).toBeDefined();
    expect(within(card).queryByText("queued")).toBeNull();

    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS));

    expect(await within(card).findByText("failed")).toBeDefined();
    expect(within(card).getByText(RUN_ERROR)).toBeDefined();
    expect(requests).toEqual(["graph-1/exec-1", "graph-1/exec-1"]);

    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 3));
    expect(requests).toHaveLength(2);
  });

  it("turns a run that succeeds after the turn to completed", async () => {
    serveRuns([makeRun("COMPLETED")]);
    renderMainChat(transcript("QUEUED"));
    const card = openRunCard();

    expect(await within(card).findByText("completed")).toBeDefined();
    expect(within(card).queryByText("queued")).toBeNull();
  });

  it("shows the error of a run that was stopped after the turn", async () => {
    serveRuns([makeRun("TERMINATED", RUN_ERROR)]);
    renderMainChat(transcript("QUEUED"));
    const card = openRunCard();

    expect(await within(card).findByText("terminated")).toBeDefined();
    expect(within(card).getByText(RUN_ERROR)).toBeDefined();
  });

  it("follows a run started through run_capability too", async () => {
    serveRuns([makeRun("FAILED", RUN_ERROR)]);
    renderMainChat(transcript("QUEUED", "run_capability"));
    const card = openRunCard();

    expect(await within(card).findByText("failed")).toBeDefined();
    expect(within(card).getByText(RUN_ERROR)).toBeDefined();
  });

  it("makes no request for a scheduled run, whose id is the schedule's", async () => {
    const requests = serveRuns([makeRun("FAILED", "should not show")]);
    renderMainChat(transcript("SCHEDULED"));
    const card = openRunCard();

    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 2));

    expect(requests).toHaveLength(0);
    expect(within(card).getByText("scheduled")).toBeDefined();
  });

  it("makes no request for a run that had already ended", async () => {
    const requests = serveRuns([makeRun("FAILED", "should not show")]);
    renderMainChat(transcript("COMPLETED"));
    const card = openRunCard();

    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 2));

    expect(requests).toHaveLength(0);
    expect(within(card).getByText("completed")).toBeDefined();
  });

  it("makes no request in the builder panel, which shows the run in place", async () => {
    const requests = serveRuns([makeRun("FAILED", "should not show")]);
    renderBuilderChat(transcript("QUEUED"));
    const card = openRunCard();

    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 2));

    expect(requests).toHaveLength(0);
    expect(within(card).getByText("queued")).toBeDefined();
  });

  it("keeps the started status when the run can't be read", async () => {
    let calls = 0;
    server.use(
      http.get(EXECUTION_URL, () => {
        calls += 1;
        return HttpResponse.json({ detail: "Not found" }, { status: 404 });
      }),
    );
    renderMainChat(transcript("QUEUED"));
    const card = openRunCard();

    await waitFor(() => expect(calls).toBe(1));
    await act(() => vi.advanceTimersByTimeAsync(LIVE_STATUS_POLL_MS * 2));

    expect(calls).toBe(1);
    expect(within(card).getByText("queued")).toBeDefined();
  });
});
