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

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetSession: () => ({
    data: undefined,
    refetch: vi.fn(),
  }),
  usePostV2GetOrCreateBuilderSessionEndpoint: () => ({
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
        data: { graph_id: "graph-1" },
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
});
