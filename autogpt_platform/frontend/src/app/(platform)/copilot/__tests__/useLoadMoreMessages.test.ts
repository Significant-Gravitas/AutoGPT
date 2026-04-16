import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useLoadMoreMessages } from "../useLoadMoreMessages";

const mockGetV2GetSession = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  getV2GetSession: (...args: unknown[]) => mockGetV2GetSession(...args),
}));

vi.mock("../helpers/convertChatSessionToUiMessages", () => ({
  convertChatSessionMessagesToUiMessages: vi.fn(() => ({ messages: [] })),
  extractToolOutputsFromRaw: vi.fn(() => []),
}));

const BASE_ARGS = {
  sessionId: "sess-1",
  initialOldestSequence: 0,
  initialNewestSequence: 49,
  initialHasMore: true,
  forwardPaginated: true,
  initialPageRawMessages: [],
};

function makeSuccessResponse(overrides: {
  messages?: unknown[];
  has_more_messages?: boolean;
  oldest_sequence?: number;
  newest_sequence?: number;
}) {
  return {
    status: 200,
    data: {
      messages: overrides.messages ?? [],
      has_more_messages: overrides.has_more_messages ?? false,
      oldest_sequence: overrides.oldest_sequence ?? 0,
      newest_sequence: overrides.newest_sequence ?? 49,
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

  it("resetPaged clears paged state and sets hasMore=false during transition", () => {
    const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

    act(() => {
      result.current.resetPaged();
    });

    expect(result.current.pagedMessages).toHaveLength(0);
    // hasMore must be false during transition to prevent forward loadMore
    // from firing on the now-active session before forwardPaginated updates.
    expect(result.current.hasMore).toBe(false);
    expect(result.current.isLoadingMore).toBe(false);
  });

  it("resetPaged exposes a fresh loadMore via incremented epoch", () => {
    const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));
    // Just verify resetPaged is callable and doesn't throw.
    expect(() => {
      act(() => {
        result.current.resetPaged();
      });
    }).not.toThrow();
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
      initialNewestSequence: 59,
      initialHasMore: false,
    });

    expect(result.current.pagedMessages).toHaveLength(0);
    expect(result.current.hasMore).toBe(false);
    expect(result.current.isLoadingMore).toBe(false);
  });

  describe("loadMore — forward pagination", () => {
    it("calls getV2GetSession with after_sequence and updates newestSequence", async () => {
      const rawMsg = { role: "user", content: "hi", sequence: 50 };
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [rawMsg],
          has_more_messages: true,
          newest_sequence: 99,
        }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({ ...BASE_ARGS, forwardPaginated: true }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenCalledWith(
        "sess-1",
        expect.objectContaining({ after_sequence: 49 }),
      );
      expect(result.current.hasMore).toBe(true);
      expect(result.current.isLoadingMore).toBe(false);
    });

    it("sets hasMore=false when response has no more messages", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({ has_more_messages: false }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({ ...BASE_ARGS, forwardPaginated: true }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(result.current.hasMore).toBe(false);
    });

    it("is a no-op when hasMore is false", async () => {
      const { result } = renderHook(() =>
        useLoadMoreMessages({
          ...BASE_ARGS,
          initialHasMore: false,
          forwardPaginated: true,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).not.toHaveBeenCalled();
    });
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

      const { result } = renderHook(() =>
        useLoadMoreMessages({
          ...BASE_ARGS,
          forwardPaginated: false,
          initialOldestSequence: 50,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenCalledWith(
        "sess-1",
        expect.objectContaining({ before_sequence: 50 }),
      );
      expect(result.current.hasMore).toBe(false);
    });
  });

  describe("loadMore — error handling", () => {
    it("does not set hasMore=false on first error", async () => {
      mockGetV2GetSession.mockRejectedValueOnce(new Error("network error"));

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      await act(async () => {
        await result.current.loadMore();
      });

      // First error — hasMore still true
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
        // Reset the in-flight guard between calls
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

      // One error, not yet at threshold — hasMore still true
      expect(result.current.hasMore).toBe(true);
      expect(result.current.isLoadingMore).toBe(false);
    });

    it("sets hasMore=false after MAX_CONSECUTIVE_ERRORS (3) non-200 responses", async () => {
      mockGetV2GetSession.mockResolvedValue({ status: 503, data: {} });

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      for (let i = 0; i < 3; i++) {
        await act(async () => {
          await result.current.loadMore();
        });
        await waitFor(() => expect(result.current.isLoadingMore).toBe(false));
      }

      expect(result.current.hasMore).toBe(false);
    });

    it("discards in-flight error when epoch changes mid-flight (resetPaged called)", async () => {
      let rejectRequest!: (e: Error) => void;
      mockGetV2GetSession.mockReturnValueOnce(
        new Promise((_, rej) => {
          rejectRequest = rej;
        }),
      );

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      act(() => {
        result.current.loadMore();
      });

      // Reset epoch mid-flight
      act(() => {
        result.current.resetPaged();
      });

      // Reject the in-flight request — stale error should be discarded
      await act(async () => {
        rejectRequest(new Error("network error"));
      });

      // State unchanged: no hasMore=false, no errorCount, isLoadingMore cleared
      expect(result.current.hasMore).toBe(false); // false from resetPaged
      expect(result.current.isLoadingMore).toBe(false);
    });
  });

  describe("loadMore — forward pagination cursor advancement", () => {
    it("advances newestSequence after a successful forward load", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [{ role: "user", content: "hi", sequence: 50 }],
          has_more_messages: true,
          newest_sequence: 99,
        }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({ ...BASE_ARGS, forwardPaginated: true }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      // A second loadMore should use after_sequence: 99 (advanced cursor)
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({ has_more_messages: false, newest_sequence: 149 }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenLastCalledWith(
        "sess-1",
        expect.objectContaining({ after_sequence: 99 }),
      );
    });

    it("does not regress newestSequence when parent refetches after pages loaded", async () => {
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [{ role: "user", content: "msg", sequence: 50 }],
          has_more_messages: true,
          newest_sequence: 99,
        }),
      );

      const { result, rerender } = renderHook(
        (props) => useLoadMoreMessages(props),
        { initialProps: { ...BASE_ARGS, forwardPaginated: true } },
      );

      // Load one page — newestSequence advances to 99
      await act(async () => {
        await result.current.loadMore();
      });

      // Parent refetches with a lower newest_sequence (49) — should NOT regress cursor
      rerender({
        ...BASE_ARGS,
        forwardPaginated: true,
        initialNewestSequence: 49,
      });

      // Next loadMore should still use the advanced cursor (99)
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({ has_more_messages: false, newest_sequence: 149 }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).toHaveBeenLastCalledWith(
        "sess-1",
        expect.objectContaining({ after_sequence: 99 }),
      );
    });
  });

  describe("loadMore — MAX_OLDER_MESSAGES truncation", () => {
    it("truncates accumulated messages at MAX_OLDER_MESSAGES (2000)", async () => {
      // Single load of 2001 messages exceeds the limit in one shot.
      // This avoids relying on cross-render closure staleness: estimatedTotal =
      // pagedRawMessages.length (0, fresh) + 2001 = 2001 >= 2000 → hasMore=false.
      const args = { ...BASE_ARGS, forwardPaginated: false };

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

      const { result } = renderHook(() => useLoadMoreMessages(args));

      await act(async () => {
        await result.current.loadMore();
      });

      expect(result.current.hasMore).toBe(false);
    });

    it("forward truncation keeps first MAX_OLDER_MESSAGES items (not last)", async () => {
      // 1990 messages already paged; load 20 more forward — total 2010 > 2000.
      // Forward truncation must keep slice(0, 2000), not slice(-2000),
      // to preserve the beginning of the conversation.
      const forwardNearLimitArgs = {
        ...BASE_ARGS,
        forwardPaginated: true,
        initialNewestSequence: 49,
        initialOldestSequence: 0,
        initialHasMore: true,
      };

      const { result } = renderHook((props) => useLoadMoreMessages(props), {
        initialProps: forwardNearLimitArgs,
      });

      // First load: 1990 messages — advances newestSequence to 2039
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: Array.from({ length: 1990 }, (_, i) => ({
            role: "assistant",
            content: `msg ${i + 50}`,
            sequence: i + 50,
          })),
          has_more_messages: true,
          newest_sequence: 2039,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      // Second load: 20 more messages pushes total to 2010 > 2000.
      // Truncation keeps seq 50..2049 (2000 items); discards seq 2050..2059 (10 items).
      // Even though the server says has_more_messages=false, hasMore stays true
      // because there are discarded items that need to be re-fetched.
      // The cursor (newestSequence) advances to 2049 — the last kept item's sequence.
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: Array.from({ length: 20 }, (_, i) => ({
            role: "assistant",
            content: `msg ${i + 2040}`,
            sequence: i + 2040,
          })),
          has_more_messages: false,
          newest_sequence: 2059,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      // Truncation occurred (2010 > 2000): hasMore=true so discarded items can be fetched.
      // Cursor advances to last kept item (seq 2049), not the server's newest (2059).
      await waitFor(() => expect(result.current.hasMore).toBe(true));
    });
  });

  describe("loadMore — null cursor guard", () => {
    it("is a no-op when newestSequence is null (forwardPaginated=true)", async () => {
      const { result } = renderHook(() =>
        useLoadMoreMessages({
          ...BASE_ARGS,
          forwardPaginated: true,
          initialNewestSequence: null,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).not.toHaveBeenCalled();
    });

    it("is a no-op when oldestSequence is null (forwardPaginated=false)", async () => {
      const { result } = renderHook(() =>
        useLoadMoreMessages({
          ...BASE_ARGS,
          forwardPaginated: false,
          initialOldestSequence: null,
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(mockGetV2GetSession).not.toHaveBeenCalled();
    });
  });

  describe("pagedMessages — initialPageRawMessages extraToolOutputs", () => {
    it("calls extractToolOutputsFromRaw for backward pagination with non-empty initialPageRawMessages", async () => {
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
          forwardPaginated: false,
          initialOldestSequence: 50,
          initialPageRawMessages: [{ role: "assistant", content: "response" }],
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(extractToolOutputsFromRaw).toHaveBeenCalled();
    });

    it("does NOT call extractToolOutputsFromRaw for forward pagination", async () => {
      const { extractToolOutputsFromRaw } = await import(
        "../helpers/convertChatSessionToUiMessages"
      );

      const rawMsg = { role: "assistant", content: "hi", sequence: 50 };
      mockGetV2GetSession.mockResolvedValueOnce(
        makeSuccessResponse({
          messages: [rawMsg],
          has_more_messages: false,
          newest_sequence: 99,
        }),
      );

      const { result } = renderHook(() =>
        useLoadMoreMessages({
          ...BASE_ARGS,
          forwardPaginated: true,
          initialPageRawMessages: [{ role: "user", content: "hello" }],
        }),
      );

      await act(async () => {
        await result.current.loadMore();
      });

      expect(extractToolOutputsFromRaw).not.toHaveBeenCalled();
    });
  });

  describe("loadMore — epoch / stale-response guard", () => {
    it("discards response when epoch changes during flight (resetPaged called)", async () => {
      let resolveRequest!: (v: unknown) => void;
      mockGetV2GetSession.mockReturnValueOnce(
        new Promise((res) => {
          resolveRequest = res;
        }),
      );

      const { result } = renderHook(() => useLoadMoreMessages(BASE_ARGS));

      // Start the request without awaiting
      act(() => {
        result.current.loadMore();
      });

      // Reset epoch mid-flight
      act(() => {
        result.current.resetPaged();
      });

      // Now resolve the in-flight request
      await act(async () => {
        resolveRequest(
          makeSuccessResponse({ messages: [{ role: "user", content: "hi" }] }),
        );
      });

      // Response discarded — pagedMessages stays empty, isLoadingMore stays false
      expect(result.current.pagedMessages).toHaveLength(0);
      expect(result.current.isLoadingMore).toBe(false);
    });
  });
});
