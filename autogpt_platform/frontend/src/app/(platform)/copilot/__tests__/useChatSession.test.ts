import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatSession } from "../useChatSession";

const mockUseGetV2GetSession = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetSession: (...args: unknown[]) => mockUseGetV2GetSession(...args),
  usePostV2CreateSession: () => ({ mutateAsync: vi.fn(), isPending: false }),
  getGetV2GetSessionQueryKey: (id: string) => ["session", id],
  getGetV2ListSessionsQueryKey: () => ["sessions"],
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: vi.fn(),
    setQueryData: vi.fn(),
  }),
}));

vi.mock("nuqs", () => ({
  parseAsString: { withDefault: (v: unknown) => v },
  useQueryState: () => ["sess-1", vi.fn()],
}));

vi.mock("../helpers/convertChatSessionToUiMessages", () => ({
  convertChatSessionMessagesToUiMessages: vi.fn(() => ({
    messages: [],
    stats: new Map(),
  })),
}));

vi.mock("../helpers", () => ({
  resolveSessionDryRun: vi.fn(() => false),
}));

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

function makeQueryResult(
  data: object | null,
  overrides: Record<string, unknown> = {},
) {
  return {
    data: data ? { status: 200, data } : undefined,
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    ...overrides,
  };
}

describe("useChatSession — pagination metadata", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns null for oldestSequence when no session data", () => {
    mockUseGetV2GetSession.mockReturnValue(makeQueryResult(null));
    const { result } = renderHook(() => useChatSession());
    expect(result.current.oldestSequence).toBeNull();
  });

  it("returns oldestSequence from session data", () => {
    mockUseGetV2GetSession.mockReturnValue(
      makeQueryResult({
        messages: [],
        has_more_messages: true,
        oldest_sequence: 50,
        active_stream: null,
      }),
    );
    const { result } = renderHook(() => useChatSession());
    expect(result.current.oldestSequence).toBe(50);
  });

  it("returns hasMoreMessages from session data", () => {
    mockUseGetV2GetSession.mockReturnValue(
      makeQueryResult({
        messages: [],
        has_more_messages: true,
        oldest_sequence: 0,
        active_stream: null,
      }),
    );
    const { result } = renderHook(() => useChatSession());
    expect(result.current.hasMoreMessages).toBe(true);
  });

  it("does not expose cached session data while a refetch is in progress", () => {
    mockUseGetV2GetSession.mockReturnValue(
      makeQueryResult(
        {
          messages: [{ role: "user", content: "stale" }],
          has_more_messages: true,
          oldest_sequence: 50,
          active_stream: true,
        },
        { isFetching: true },
      ),
    );

    const { result } = renderHook(() => useChatSession());

    expect(result.current.hydratedMessages).toBeUndefined();
    expect(result.current.rawSessionMessages).toEqual([]);
    expect(result.current.hasActiveStream).toBe(false);
    expect(result.current.hasMoreMessages).toBe(false);
    expect(result.current.oldestSequence).toBeNull();
    // isLoadingSession is scoped to the INITIAL fetch only (isLoading) so
    // background refetches do not disable the chat input between keystrokes.
    expect(result.current.isLoadingSession).toBe(false);
  });

  it("treats the initial fetch as a loading session", () => {
    mockUseGetV2GetSession.mockReturnValue(
      makeQueryResult(null, { isLoading: true, isFetching: true }),
    );

    const { result } = renderHook(() => useChatSession());

    expect(result.current.isLoadingSession).toBe(true);
  });
});
