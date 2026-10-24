import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useFollowQueuedTurn } from "./useFollowQueuedTurn";

describe("useFollowQueuedTurn", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  function renderQueued(enabled = true) {
    const refetchSession = vi.fn(async () => undefined);
    const followBackendTurn = vi.fn();
    const view = renderHook(
      ({ sessionChatStatus }: { sessionChatStatus: string | undefined }) =>
        useFollowQueuedTurn({
          enabled,
          sessionChatStatus,
          refetchSession,
          followBackendTurn,
        }),
      {
        initialProps: {
          sessionChatStatus: "queued" as string | undefined,
        },
      },
    );
    return { view, refetchSession, followBackendTurn };
  }

  it("re-reads a session queued at the running cap until its turn starts", () => {
    const { refetchSession, followBackendTurn } = renderQueued();

    vi.advanceTimersByTime(30_000);

    expect(refetchSession).toHaveBeenCalledTimes(3);
    expect(followBackendTurn).not.toHaveBeenCalled();
  });

  it("follows the turn once the session leaves the queue, and stops reading", () => {
    const { view, refetchSession, followBackendTurn } = renderQueued();

    view.rerender({ sessionChatStatus: "running" });
    vi.advanceTimersByTime(30_000);

    expect(followBackendTurn).toHaveBeenCalledTimes(1);
    expect(refetchSession).not.toHaveBeenCalled();
  });

  it("keeps waiting while a re-read of the still-queued session is in flight", () => {
    const { view, refetchSession, followBackendTurn } = renderQueued();

    // The session hook reports no status while its query refetches.
    view.rerender({ sessionChatStatus: undefined });
    view.rerender({ sessionChatStatus: "queued" });
    vi.advanceTimersByTime(10_000);

    expect(followBackendTurn).not.toHaveBeenCalled();
    expect(refetchSession).toHaveBeenCalledTimes(1);
  });

  it("leaves a session that was never queued alone", () => {
    const refetchSession = vi.fn(async () => undefined);
    const followBackendTurn = vi.fn();
    const view = renderHook(
      ({ sessionChatStatus }: { sessionChatStatus: string }) =>
        useFollowQueuedTurn({
          enabled: true,
          sessionChatStatus,
          refetchSession,
          followBackendTurn,
        }),
      { initialProps: { sessionChatStatus: "idle" } },
    );

    view.rerender({ sessionChatStatus: "running" });
    vi.advanceTimersByTime(30_000);

    expect(refetchSession).not.toHaveBeenCalled();
    expect(followBackendTurn).not.toHaveBeenCalled();
  });

  it("does nothing where the stream runtime follows the queue itself", () => {
    const { refetchSession } = renderQueued(false);

    vi.advanceTimersByTime(30_000);

    expect(refetchSession).not.toHaveBeenCalled();
  });
});
