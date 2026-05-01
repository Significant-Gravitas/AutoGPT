import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, renderHook } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getGetV2ListSessionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { useSessionTitlePoll } from "../useSessionTitlePoll";

interface SessionRow {
  id: string;
  title: string | null;
}

function listKey() {
  return getGetV2ListSessionsQueryKey({ limit: 50 });
}

function seedSessions(client: QueryClient, sessions: SessionRow[]) {
  client.setQueryData(listKey(), { status: 200, data: { sessions } });
}

function makeWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children);
  };
}

interface RenderProps {
  status: "submitted" | "streaming" | "ready" | "error";
  sessionId: string | null;
  isReconnecting: boolean;
}

function renderPollHook(client: QueryClient, initial: RenderProps) {
  return renderHook(
    ({ status, sessionId, isReconnecting }: RenderProps) =>
      useSessionTitlePoll({ status, sessionId, isReconnecting }),
    {
      wrapper: makeWrapper(client),
      initialProps: initial,
    },
  );
}

describe("useSessionTitlePoll", () => {
  let client: QueryClient;
  let invalidateSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.useFakeTimers();
    client = new QueryClient();
    invalidateSpy = vi.spyOn(client, "invalidateQueries");
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("does not poll when there is no session", () => {
    renderPollHook(client, {
      status: "ready",
      sessionId: null,
      isReconnecting: false,
    });
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("does not poll when reconnecting", async () => {
    const { rerender } = renderPollHook(client, {
      status: "streaming",
      sessionId: "s1",
      isReconnecting: true,
    });
    rerender({ status: "ready", sessionId: "s1", isReconnecting: true });
    await vi.advanceTimersByTimeAsync(0);
    expect(invalidateSpy).not.toHaveBeenCalled();
  });

  it("kicks off an immediate invalidate + interval on streaming→ready", async () => {
    seedSessions(client, [{ id: "s1", title: null }]);
    const { rerender } = renderPollHook(client, {
      status: "streaming",
      sessionId: "s1",
      isReconnecting: false,
    });
    rerender({ status: "ready", sessionId: "s1", isReconnecting: false });

    // Immediate invalidate.
    expect(invalidateSpy).toHaveBeenCalledTimes(1);

    // Each tick re-invalidates while no title appears.
    await vi.advanceTimersByTimeAsync(2_000);
    expect(invalidateSpy).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(invalidateSpy).toHaveBeenCalledTimes(3);
  });

  it("stops polling once a title shows up in the cached session list", async () => {
    seedSessions(client, [{ id: "s1", title: null }]);
    const { rerender } = renderPollHook(client, {
      status: "streaming",
      sessionId: "s1",
      isReconnecting: false,
    });
    rerender({ status: "ready", sessionId: "s1", isReconnecting: false });
    expect(invalidateSpy).toHaveBeenCalledTimes(1); // initial

    // Backend writes the title before the next tick.
    seedSessions(client, [{ id: "s1", title: "New title" }]);
    await vi.advanceTimersByTimeAsync(2_000);
    // First tick: clears interval (sees title), no extra invalidate.
    expect(invalidateSpy).toHaveBeenCalledTimes(1);

    // Subsequent ticks must NOT fire — interval cleared.
    await vi.advanceTimersByTimeAsync(20_000);
    expect(invalidateSpy).toHaveBeenCalledTimes(1);
  });

  it("gives up after the max attempt count", async () => {
    seedSessions(client, [{ id: "s1", title: null }]);
    const { rerender } = renderPollHook(client, {
      status: "streaming",
      sessionId: "s1",
      isReconnecting: false,
    });
    rerender({ status: "ready", sessionId: "s1", isReconnecting: false });
    // Initial + 5 attempts = 6 total invalidations, then nothing more.
    for (let i = 0; i < 5; i++) await vi.advanceTimersByTimeAsync(2_000);
    expect(invalidateSpy).toHaveBeenCalledTimes(6);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(invalidateSpy).toHaveBeenCalledTimes(6);
  });

  it("clears the interval when sessionId changes", async () => {
    seedSessions(client, [{ id: "s1", title: null }]);
    const { rerender } = renderPollHook(client, {
      status: "streaming",
      sessionId: "s1",
      isReconnecting: false,
    });
    rerender({ status: "ready", sessionId: "s1", isReconnecting: false });
    expect(invalidateSpy).toHaveBeenCalledTimes(1); // initial for s1

    rerender({ status: "ready", sessionId: "s2", isReconnecting: false });
    invalidateSpy.mockClear();
    // The s1 interval should be torn down — no further invalidates from it.
    await vi.advanceTimersByTimeAsync(20_000);
    expect(invalidateSpy).not.toHaveBeenCalled();
  });
});
