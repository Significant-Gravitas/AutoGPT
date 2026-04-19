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
  mockUseQueryStates.mockReturnValue([
    { flowID: null, flowExecutionID: null, flowVersion: null },
    vi.fn(),
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
});
