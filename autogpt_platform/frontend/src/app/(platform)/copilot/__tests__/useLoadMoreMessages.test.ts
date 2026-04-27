import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useLoadMoreMessages } from "../useLoadMoreMessages";

const mockGetV2GetSession = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  getV2GetSession: (...args: unknown[]) => mockGetV2GetSession(...args),
}));

vi.mock("../helpers/convertChatSessionToUiMessages", () => ({
  convertChatSessionMessagesToUiMessages: vi.fn(() => ({
    messages: [],
    stats: new Map(),
  })),
  extractToolOutputsFromRaw: vi.fn(() => []),
}));

const BASE_ARGS = {
  sessionId: "sess-1",
  initialOldestSequence: 50,
  initialHasMore: true,
  initialPageRawMessages: [],
};

function makeSuccessResponse(overrides: {
  messages?: unknown[];
  has_more_messages?: boolean;
  oldest_sequence?: number;
}) {
  return {
    status: 200,
    data: {
      messages: overrides.messages ?? [],
      has_more_messages: overrides.has_more_messages ?? false,
      oldest_sequence: overrides.oldest_sequence ?? 0,
    },
  };
}

describe("useLoadMoreMessages", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("initialises with empty pagedMessages and correct cursors", () => {
    const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));
    expect(result.current.pagedMessages).toHaveLength(0);
    expect(result.current.hasMore).toBe(true);
    expect(result.current.isLoadingMore).toBe(false);
  });

  it("resets all state on sessionId change", () => {
    const { result, rerender } = renderHook(
      (props) => useLoadMoreMessages(props),
      { initialProps: BASE_ARGS },
    );

    rerender({
      ...BASE_ARGS,
      sessionId: "sess-2",
      initialOldestSequence: 10,
      initialHasMore: false,
    });

    expect(result.current.pagedMessages).toHaveLength(0);
    expect(result.current.hasMore).toBe(false);
    expect(result.current.isLoadingMore).toBe(false);
  });

  describe("loadMore — backward pagination", () => {
    it("calls getV2GetSession with before_sequence", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [{ role: "user", content: "old", sequence: 0 }],
          has_more_messages: false,
          oldest_sequence: 0,
        }),
      );

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenCalledWith(
        "sess-1",
        expect.objectContaining({ before_sequence: 50 }),
      );
      expect(result.current.hasMore).toBe(false);
    });

    it("is a no-op when hasMore is false", async () => {
      const { result } = renderHook(() =>
        useLoadMoreMessages({ ...BASE_ARGS, initialHasMore: false }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).not.toHaveBeenCalled();
    });

    it("is a no-op when oldestSequence is null", async () => {
      const { result } = renderHook(() =>
        useLoadMoreMessages({ ...BASE_ARGS, initialOldestSequence: null }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).not.toHaveBeenCalled();
    });
  });

  describe("loadMore — error handling", () => {
    it("does not set hasMore=false on first error", async () => {
      mockGetV2GetSession.mockRejectedValueOnce(new Error("network error"));

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      await act(async () => {
        await result.current.loadMore();
      });

      expect(result.current.hasMore).toBe(true);
      expect(result.current.isLoadingMore).toBe(false);
    });

    it("sets hasMore=false after MAX_CONSECUTIVE_ERRORS (3) errors", async () => {
      mockGetV2GetSession.mockRejectedValue(new Error("network error"));

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      for (let i = 0; i < 3; i++) {
        await act(async () => {
          await result.current.loadMore();
        });
        await waitFor(() => expect(result.current.isLoadingMore).toBe(false));
      }

      expect(result.current.hasMore).toBe(false);
    });

    it("ignores non-200 response and increments error count", async () => {
      mockGetV2GetSession.mockResolvedValueOnce({ status: 500, data: {} });

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      await act(async () => {
        await result.current.loadMore();
      });

      expect(result.current.hasMore).toBe(true);
      expect(result.current.isLoadingMore).toBe(false);
    });
  });

  describe("loadMore — MAX_OLDER_MESSAGES truncation", () => {
    it("truncates accumulated messages at MAX_OLDER_MESSAGES (2000)", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: Array.from({ length: 2001 }, (_, i) => ({
            role: "user",
            content: `msg ${i}`,
            sequence: i,
          })),
          has_more_messages: true,
          oldest_sequence: 0,
        }),
      );

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      await act(async () => {
        await result.current.loadMore();
      });

      expect(result.current.hasMore).toBe(false);
    });
  });

  describe("pagedMessages — initialPageRawMessages extraToolOutputs", () => {
    it("calls extractToolOutputsFromRaw with non-empty initialPageRawMessages", async () => {
      const { extractToolOutputsFromRaw } = await import(
        "../helpers/convertChatSessionToUiMessages"
      );

      const rawMsg = { role: "user", content: "old", sequence: 0 };
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [rawMsg],
          has_more_messages: false,
          oldest_sequence: 0,
        }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({
          ...BASE_ARGS,
          initialPageRawMessages: [{ role: "assistant", content: "response" }],
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(extractToolOutputsFromRaw).toHaveBeenCalled();
    });
  });
});
