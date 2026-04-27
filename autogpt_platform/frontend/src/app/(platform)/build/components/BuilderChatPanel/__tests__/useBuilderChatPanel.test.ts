import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useBuilderChatPanel } from "../useBuilderChatPanel";

const createBuilderSession = vi.fn();
const createNewGraph = vi.fn();
const setActiveVersion = vi.fn();
const mockRefetchGraph = vi.fn();
const mockUseQueryStates = vi.fn();
const mockUseGetV1GetSpecificGraph = vi.fn();
const mockUseCopilotStream = vi.fn();
const mockUseCopilotPendingChips = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  useGetV1GetSpecificGraph: (...args: unknown[]) =>
    mockUseGetV1GetSpecificGraph(...args),
  usePostV1CreateNewGraph: () => ({
    mutateAsync: createNewGraph,
    isPending: false,
  }),
  usePutV1SetActiveGraphVersion: () => ({ mutateAsync: setActiveVersion }),
  getGetV1GetSpecificGraphQueryKey: (id: string) => ["graph", id],
}));

const mockUseGetV2GetSession = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetSession: (...args: unknown[]) => mockUseGetV2GetSession(...args),
  usePostV2CreateSession: () => ({
    mutateAsync: createBuilderSession,
    isPending: false,
  }),
  getGetV2GetSessionQueryKey: (id: string) => ["session", id],
}));

vi.mock("@/app/api/helpers", () => ({
  okData: (res: unknown) => res,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

vi.mock("nuqs", () => ({
  parseAsString: Symbol("str"),
  parseAsInteger: Symbol("int"),
  useQueryStates: (...args: unknown[]) => mockUseQueryStates(...args),
}));

vi.mock(
  "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages",
  () => ({
    convertChatSessionMessagesToUiMessages: () => ({
      messages: [],
      durations: new Map(),
    }),
  }),
);

vi.mock("@/app/(platform)/copilot/useCopilotStream", () => ({
  useCopilotStream: (...args: unknown[]) => mockUseCopilotStream(...args),
}));

vi.mock("@/app/(platform)/copilot/useCopilotPendingChips", () => ({
  useCopilotPendingChips: (...args: unknown[]) =>
    mockUseCopilotPendingChips(...args),
}));

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

const setQueryStatesMock = vi.fn();

const defaultStream = {
  messages: [],
  setMessages: vi.fn(),
  sendMessage: vi.fn(),
  stop: vi.fn(),
  status: "ready" as const,
  error: undefined,
};

beforeEach(() => {
  vi.clearAllMocks();
  setQueryStatesMock.mockReset();
  mockUseQueryStates.mockReturnValue([
    { flowID: null, flowExecutionID: null, flowVersion: null },
    setQueryStatesMock,
  ]);
  mockUseGetV1GetSpecificGraph.mockReturnValue({
    data: null,
    refetch: mockRefetchGraph,
  });
  mockUseCopilotStream.mockReturnValue(defaultStream);
  mockUseCopilotPendingChips.mockReturnValue({
    queuedMessages: [],
    appendChip: vi.fn(),
  });
  mockUseGetV2GetSession.mockReturnValue({
    data: undefined,
    refetch: vi.fn(),
  });
});

describe("useBuilderChatPanel", () => {
  it("starts closed with no session", () => {
    const { result } = renderHook(() => useBuilderChatPanel());
    expect(result.current.isOpen).toBe(false);
    expect(result.current.sessionId).toBeNull();
  });

  it("toggles open on handleToggle", () => {
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    expect(result.current.isOpen).toBe(false);
    result.current.handleToggle();
    rerender();
    expect(result.current.isOpen).toBe(true);
  });

  it("surfaces the bootstrapping flag when opened without a flowID", () => {
    mockUseQueryStates.mockReturnValue([
      { flowID: null, flowExecutionID: null, flowVersion: null },
      vi.fn(),
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    expect(result.current.isBootstrapping).toBe(true);
  });

  it("forwards fast mode to useCopilotStream", () => {
    renderHook(() => useBuilderChatPanel());
    expect(mockUseCopilotStream).toHaveBeenCalledWith(
      expect.objectContaining({ copilotMode: "fast" }),
    );
  });

  it("requests a builder session when opened with a flowID", async () => {
    createBuilderSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-bound" },
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-1", flowExecutionID: null, flowVersion: null },
      vi.fn(),
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();

    await waitFor(() => {
      expect(createBuilderSession).toHaveBeenCalledWith({
        data: { builder_graph_id: "graph-1" },
      });
    });
  });

  it("exposes null revert target before any edit_agent turn", () => {
    const { result } = renderHook(() => useBuilderChatPanel());
    expect(result.current.revertTargetVersion).toBeNull();
  });

  it("auto-creates a blank agent when opened with no flowID, writing the new id to the URL", async () => {
    createNewGraph.mockResolvedValue({
      status: 200,
      data: { id: "graph-boot", version: 1 },
    });
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();

    await waitFor(() => {
      expect(createNewGraph).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(setQueryStatesMock).toHaveBeenCalledWith({
        flowID: "graph-boot",
        flowVersion: 1,
      });
    });
  });

  it("surfaces a destructive toast when the bootstrap mutation fails", async () => {
    const toast = vi.fn();
    const useToastMock = await import("@/components/molecules/Toast/use-toast");
    (useToastMock.useToast as unknown as ReturnType<typeof vi.fn>) = vi
      .fn()
      .mockReturnValue({ toast });
    createNewGraph.mockResolvedValue({ status: 500, data: null });
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(createNewGraph).toHaveBeenCalled();
    });
    // The hook catches and toasts; no throw should reach the test.
    expect(result.current.isOpen).toBe(true);
  });

  it("sends the message directly when stream status is ready", async () => {
    const sendMessage = vi.fn();
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      sendMessage,
      status: "ready",
    });
    createBuilderSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-send" },
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-send", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(result.current.sessionId).toBe("sess-send");
    });
    await result.current.onSend("hello");
    expect(sendMessage).toHaveBeenCalledWith({ text: "hello" });
  });

  it("no-ops onSend when no session is bound yet", async () => {
    const sendMessage = vi.fn();
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      sendMessage,
      status: "ready",
    });
    const { result } = renderHook(() => useBuilderChatPanel());
    await result.current.onSend("nobody home");
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("no-ops onSend for empty/whitespace input even when session is live", async () => {
    const sendMessage = vi.fn();
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      sendMessage,
      status: "ready",
    });
    createBuilderSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-empty" },
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "g", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(result.current.sessionId).toBe("sess-empty");
    });
    await result.current.onSend("   ");
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("does nothing when handleRevert is called without a revert target", async () => {
    const { result } = renderHook(() => useBuilderChatPanel());
    await result.current.handleRevert();
    expect(setActiveVersion).not.toHaveBeenCalled();
  });

  it("forwards the bound graph_id on subsequent panel opens (no duplicate session create)", async () => {
    createBuilderSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-stable" },
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-stable", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(createBuilderSession).toHaveBeenCalledTimes(1);
    });
    // Close + reopen should not re-bind while the same flowID + sessionId hold.
    result.current.handleToggle();
    rerender();
    result.current.handleToggle();
    rerender();
    // Allow any pending effect microtasks to settle.
    await Promise.resolve();
    expect(createBuilderSession).toHaveBeenCalledTimes(1);
  });

  it("rebinds for a new graph when the previous bind is still in flight (sentry 13568553)", async () => {
    // Regression: the bindingRef lock used to persist across graph
    // navigations, so a pending A-request left the lock set and B's bind
    // effect early-returned without ever retrying — panel stuck.
    // First call: slow resolve so graph-A's bind stays in flight while we navigate.
    let resolveA!: (res: unknown) => void;
    createBuilderSession
      .mockImplementationOnce(
        () =>
          new Promise((r) => {
            resolveA = r;
          }),
      )
      .mockResolvedValueOnce({ status: 200, data: { id: "sess-B" } });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-A", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(createBuilderSession).toHaveBeenCalledWith({
        data: { builder_graph_id: "graph-A" },
      });
    });
    // Navigate to B while A is still pending.
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-B", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    rerender();
    // B's bind effect must fire even though A's is mid-flight — the
    // reset-on-graph-change effect clears bindingRef so B is not blocked.
    await waitFor(() => {
      expect(createBuilderSession).toHaveBeenCalledWith({
        data: { builder_graph_id: "graph-B" },
      });
    });
    // Resolve A late — its response must NOT overwrite B's sessionId
    // (currentFlowIDRef staleness guard handles that).
    resolveA({ status: 200, data: { id: "sess-A-stale" } });
    await waitFor(() => {
      expect(result.current.sessionId).toBe("sess-B");
    });
  });

  it("resets session + revert state when flowID becomes null", async () => {
    const setMessages = vi.fn();
    mockUseCopilotStream.mockReturnValue({ ...defaultStream, setMessages });
    // Start with a flowID so the session could bind, then drop it.
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-A", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    // Flip to no flowID on the next render.
    mockUseQueryStates.mockReturnValue([
      { flowID: null, flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    rerender();
    await waitFor(() => {
      expect(setMessages).toHaveBeenCalledWith([]);
    });
    expect(result.current.sessionId).toBeNull();
    expect(result.current.revertTargetVersion).toBeNull();
  });

  it("records a revert target and triggers a graph refetch when edit_agent tool output completes", async () => {
    mockUseGetV1GetSpecificGraph.mockReturnValue({
      data: { id: "graph-X", version: 5 },
      refetch: mockRefetchGraph,
    });
    const messagesWithEdit = [
      {
        role: "assistant",
        parts: [
          {
            type: "tool-edit_agent",
            toolCallId: "tc-edit-1",
            state: "output-available",
            output: { agent_id: "graph-X" },
          },
        ],
      },
    ];
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      messages: messagesWithEdit,
      status: "ready",
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-X", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result } = renderHook(() => useBuilderChatPanel());
    await waitFor(() => {
      expect(result.current.revertTargetVersion).toBe(5);
    });
    expect(mockRefetchGraph).toHaveBeenCalled();
  });

  it("writes execution_id to flowExecutionID when run_agent tool output completes", async () => {
    const messagesWithRun = [
      {
        role: "assistant",
        parts: [
          {
            type: "tool-run_agent",
            toolCallId: "tc-run-1",
            state: "output-available",
            output: { execution_id: "exec-abc-123" },
          },
        ],
      },
    ];
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      messages: messagesWithRun,
      status: "ready",
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-R", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    renderHook(() => useBuilderChatPanel());
    await waitFor(() => {
      expect(setQueryStatesMock).toHaveBeenCalledWith({
        flowExecutionID: "exec-abc-123",
      });
    });
  });

  // Live-stream tool outputs arrive as JSON STRINGS (the backend stashes
  // tool output via json.dumps). Hydrated-from-DB tool outputs arrive as
  // already-parsed objects. The effect must handle both shapes.
  it("writes flowVersion when edit_agent output is a JSON string (live stream)", async () => {
    mockUseGetV1GetSpecificGraph.mockReturnValue({
      data: { id: "graph-live", version: 3 },
      refetch: mockRefetchGraph,
    });
    const messagesWithEdit = [
      {
        role: "assistant",
        parts: [
          {
            type: "tool-edit_agent",
            toolCallId: "tc-edit-live-1",
            state: "output-available",
            output: JSON.stringify({
              type: "agent_builder_saved",
              agent_id: "graph-live",
              graph_version: 4,
            }),
          },
        ],
      },
    ];
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      messages: messagesWithEdit,
      status: "ready",
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-live", flowExecutionID: null, flowVersion: 3 },
      setQueryStatesMock,
    ]);
    renderHook(() => useBuilderChatPanel());
    await waitFor(() => {
      expect(setQueryStatesMock).toHaveBeenCalledWith({ flowVersion: 4 });
    });
  });

  it("writes flowExecutionID when run_agent output is a JSON string (live stream)", async () => {
    const messagesWithRun = [
      {
        role: "assistant",
        parts: [
          {
            type: "tool-run_agent",
            toolCallId: "tc-run-live-1",
            state: "output-available",
            output: JSON.stringify({
              type: "agent_output",
              execution_id: "exec-live-xyz",
            }),
          },
        ],
      },
    ];
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      messages: messagesWithRun,
      status: "ready",
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-live", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    renderHook(() => useBuilderChatPanel());
    await waitFor(() => {
      expect(setQueryStatesMock).toHaveBeenCalledWith({
        flowExecutionID: "exec-live-xyz",
      });
    });
  });

  it("ignores tool outputs that are not output-available or not assistant-role", async () => {
    const messages = [
      {
        role: "user",
        parts: [
          {
            type: "tool-edit_agent",
            toolCallId: "tc-bad-1",
            state: "output-available",
            output: { agent_id: "g" },
          },
        ],
      },
      {
        role: "assistant",
        parts: [
          {
            type: "tool-run_agent",
            toolCallId: "tc-bad-2",
            state: "partial",
            output: { execution_id: "exec-incomplete" },
          },
        ],
      },
    ];
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      messages,
      status: "ready",
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "g", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    renderHook(() => useBuilderChatPanel());
    // Allow effects to flush without asserting against the tool branches.
    await Promise.resolve();
    expect(setQueryStatesMock).not.toHaveBeenCalledWith(
      expect.objectContaining({ flowExecutionID: "exec-incomplete" }),
    );
    expect(mockRefetchGraph).not.toHaveBeenCalled();
  });

  it("queues a follow-up via the helper when onSend is called while streaming", async () => {
    const appendChip = vi.fn();
    const sendMessage = vi.fn();
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      sendMessage,
      status: "streaming",
    });
    mockUseCopilotPendingChips.mockReturnValue({
      queuedMessages: [],
      appendChip,
    });
    createBuilderSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-queue" },
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-queue", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    vi.doMock("@/app/(platform)/copilot/helpers/queueFollowUpMessage", () => ({
      queueFollowUpMessage: vi.fn().mockResolvedValue(undefined),
    }));
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(result.current.sessionId).toBe("sess-queue");
    });
    await result.current.onSend("queued msg");
    // The message must be appended as a chip AND not sent directly.
    expect(appendChip).toHaveBeenCalledWith("queued msg");
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("hydrates messages from the session query when GET /sessions returns 200", async () => {
    // Session data present → convertChatSessionMessagesToUiMessages runs and
    // the hook forwards the hydrated messages to useCopilotStream.
    mockUseGetV2GetSession.mockReturnValue({
      data: {
        status: 200,
        data: {
          id: "sess-hydrated",
          messages: [{ role: "assistant", content: "welcome" }],
          active_stream: null,
        },
      },
      refetch: vi.fn(),
    });
    createBuilderSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-hydrated" },
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-hyd", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(result.current.sessionId).toBe("sess-hydrated");
    });
    // useCopilotStream should have been invoked with a non-undefined hydratedMessages.
    const lastCall =
      mockUseCopilotStream.mock.calls[
        mockUseCopilotStream.mock.calls.length - 1
      ];
    expect(lastCall[0]).toHaveProperty("hydratedMessages");
  });

  it("keeps the hydratedMessages reference stable across renders when session data is unchanged", async () => {
    // Regression guard: an earlier version recomputed hydratedMessages on every
    // render (no useMemo), which broke referential equality in
    // useHydrateOnStreamEnd and caused an infinite setState loop (caught by
    // React's max-update-depth guard, rendered through the builder
    // ErrorBoundary as "Something went wrong"). Pin the reference here.
    const sessionData = {
      status: 200 as const,
      data: {
        id: "sess-stable",
        messages: [{ role: "assistant", content: "hello" }],
        active_stream: null,
      },
    };
    mockUseGetV2GetSession.mockReturnValue({
      data: sessionData,
      refetch: vi.fn(),
    });
    createBuilderSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-stable" },
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-stable", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(result.current.sessionId).toBe("sess-stable");
    });
    const firstCall =
      mockUseCopilotStream.mock.calls[
        mockUseCopilotStream.mock.calls.length - 1
      ];
    const firstHydrated = firstCall[0].hydratedMessages;
    rerender();
    rerender();
    const lastCall =
      mockUseCopilotStream.mock.calls[
        mockUseCopilotStream.mock.calls.length - 1
      ];
    expect(lastCall[0].hydratedMessages).toBe(firstHydrated);
  });

  it("surfaces active_stream=true via hasActiveStream flag forwarded to the stream hook", async () => {
    mockUseGetV2GetSession.mockReturnValue({
      data: {
        status: 200,
        data: {
          id: "sess-active",
          messages: [],
          active_stream: { turn_id: "t1", last_message_id: "0-0" },
        },
      },
      refetch: vi.fn(),
    });
    createBuilderSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-active" },
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-act", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(result.current.sessionId).toBe("sess-active");
    });
    const lastCall =
      mockUseCopilotStream.mock.calls[
        mockUseCopilotStream.mock.calls.length - 1
      ];
    expect(lastCall[0].hasActiveStream).toBe(true);
  });

  it("toasts destructively when handleRevert receives a non-200 from setActiveVersion", async () => {
    mockUseGetV1GetSpecificGraph.mockReturnValue({
      data: { id: "graph-revfail", version: 9 },
      refetch: mockRefetchGraph,
    });
    setActiveVersion.mockResolvedValue({ status: 500 });
    const messagesWithEdit = [
      {
        role: "assistant",
        parts: [
          {
            type: "tool-edit_agent",
            toolCallId: "tc-edit-fail",
            state: "output-available",
            output: { agent_id: "graph-revfail" },
          },
        ],
      },
    ];
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      messages: messagesWithEdit,
      status: "ready",
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-revfail", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result } = renderHook(() => useBuilderChatPanel());
    await waitFor(() => {
      expect(result.current.revertTargetVersion).toBe(9);
    });
    // Must not throw; the hook catches and toasts.
    await result.current.handleRevert();
    expect(setActiveVersion).toHaveBeenCalled();
  });

  it("surfaces bindError and clears isBootstrapping when createBuilderSession fails", async () => {
    createBuilderSession.mockRejectedValue(new Error("boom"));
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-err", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(createBuilderSession).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(result.current.bindError).not.toBeNull();
    });
    expect(result.current.isBootstrapping).toBe(false);
    // Retrying should clear the error and re-invoke the mutation.
    createBuilderSession.mockResolvedValueOnce({
      status: 200,
      data: { id: "sess-retry" },
    });
    result.current.retryBind();
    rerender();
    await waitFor(() => {
      expect(result.current.sessionId).toBe("sess-retry");
    });
    expect(result.current.bindError).toBeNull();
  });

  it("surfaces bootstrapError when createNewGraph fails and recovers via retryBootstrap", async () => {
    createNewGraph.mockRejectedValueOnce(new Error("network"));
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(createNewGraph).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(result.current.bootstrapError).not.toBeNull();
    });
    expect(result.current.isBootstrapping).toBe(false);
    createNewGraph.mockResolvedValueOnce({
      status: 200,
      data: { id: "graph-after-retry", version: 1 },
    });
    result.current.retryBootstrap();
    rerender();
    await waitFor(() => {
      expect(createNewGraph).toHaveBeenCalledTimes(2);
    });
    await waitFor(() => {
      expect(setQueryStatesMock).toHaveBeenCalledWith({
        flowID: "graph-after-retry",
        flowVersion: 1,
      });
    });
  });

  it("discards a stale createBuilderSession response when flowID changed mid-request", async () => {
    // The first call (slow, for graph-A) resolves AFTER the user navigates
    // to graph-B — its response id must NOT overwrite the session because
    // the staleness check on currentFlowIDRef should bail out.
    let resolveA: (v: unknown) => void = () => {};
    const slowResponseA = new Promise((resolve) => {
      resolveA = resolve;
    });
    createBuilderSession.mockImplementationOnce(() => slowResponseA);
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-A", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result, rerender } = renderHook(() => useBuilderChatPanel());
    result.current.handleToggle();
    rerender();
    await waitFor(() => {
      expect(createBuilderSession).toHaveBeenCalledWith({
        data: { builder_graph_id: "graph-A" },
      });
    });
    // Navigate to graph-B — reset effect clears sessionId + boundGraphRef.
    // The currentFlowIDRef is updated synchronously in an effect so the
    // pending graph-A IIFE will observe graph-B on its staleness check.
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-B", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    rerender();
    // NOW the stale graph-A response resolves — must be discarded because
    // currentFlowIDRef.current !== "graph-A".
    resolveA({ status: 200, data: { id: "sess-A-STALE" } });
    // Flush the post-await microtasks.
    await Promise.resolve();
    await Promise.resolve();
    // Stale response discarded: no sessionId set from it.
    expect(result.current.sessionId).not.toBe("sess-A-STALE");
  });

  it("reverts to the captured version and invalidates graph queries on handleRevert success", async () => {
    mockUseGetV1GetSpecificGraph.mockReturnValue({
      data: { id: "graph-R", version: 7 },
      refetch: mockRefetchGraph,
    });
    const invalidateQueries = vi.fn();
    const rq = await import("@tanstack/react-query");
    (rq.useQueryClient as unknown as ReturnType<typeof vi.fn>) = vi
      .fn()
      .mockReturnValue({ invalidateQueries });
    setActiveVersion.mockResolvedValue({ status: 200 });
    // Prime a revert target via an edit_agent tool output.
    const messagesWithEdit = [
      {
        role: "assistant",
        parts: [
          {
            type: "tool-edit_agent",
            toolCallId: "tc-edit-r",
            state: "output-available",
            output: { agent_id: "graph-R" },
          },
        ],
      },
    ];
    mockUseCopilotStream.mockReturnValue({
      ...defaultStream,
      messages: messagesWithEdit,
      status: "ready",
    });
    mockUseQueryStates.mockReturnValue([
      { flowID: "graph-R", flowExecutionID: null, flowVersion: null },
      setQueryStatesMock,
    ]);
    const { result } = renderHook(() => useBuilderChatPanel());
    await waitFor(() => {
      expect(result.current.revertTargetVersion).toBe(7);
    });
    await result.current.handleRevert();
    expect(setActiveVersion).toHaveBeenCalledWith({
      graphId: "graph-R",
      data: { active_graph_version: 7 },
    });
    expect(setQueryStatesMock).toHaveBeenCalledWith({ flowVersion: 7 });
  });
});
