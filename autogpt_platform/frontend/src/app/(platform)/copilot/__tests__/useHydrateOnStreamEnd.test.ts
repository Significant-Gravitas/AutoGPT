import { renderHook } from "@testing-library/react";
import type { UIMessage } from "ai";
import { describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

import { useHydrateOnStreamEnd } from "../useHydrateOnStreamEnd";

/** Use distinct default text per id so the assistant dedup doesn't collapse them. */
function msg(id: string, text?: string): UIMessage {
  return {
    id,
    role: "assistant",
    parts: [
      {
        type: "text" as const,
        text: text ?? `text-${id}`,
        state: "done" as const,
      },
    ],
  };
}

describe("useHydrateOnStreamEnd", () => {
  it("does nothing while streaming", () => {
    const setMessages = vi.fn();
    renderHook(() =>
      useHydrateOnStreamEnd({
        status: "streaming",
        hydratedMessages: [msg("a")],
        isReconnectScheduled: false,
        hasActiveStream: false,
        setMessages,
      }),
    );
    expect(setMessages).not.toHaveBeenCalled();
  });

  it("does nothing while a reconnect is scheduled", () => {
    const setMessages = vi.fn();
    renderHook(() =>
      useHydrateOnStreamEnd({
        status: "ready",
        hydratedMessages: [msg("a")],
        isReconnectScheduled: true,
        hasActiveStream: false,
        setMessages,
      }),
    );
    expect(setMessages).not.toHaveBeenCalled();
  });

  it("length-gated top-up when there is no pending force-hydrate", () => {
    // Fresh mount at "ready" → force-hydrate was never armed.
    // The hook should top up only if the fresh data is strictly larger.
    const setMessages = vi.fn();
    const fresh = [msg("a"), msg("b"), msg("c")];
    renderHook(() =>
      useHydrateOnStreamEnd({
        status: "ready",
        hydratedMessages: fresh,
        isReconnectScheduled: false,
        hasActiveStream: false,
        setMessages,
      }),
    );
    expect(setMessages).toHaveBeenCalledTimes(1);
    // setMessages was called with an updater — invoke with a short prev.
    const updater = setMessages.mock.calls[0][0] as (
      prev: UIMessage[],
    ) => UIMessage[];
    expect(updater([msg("x")])).toHaveLength(3);
    // And with a longer prev, no-op.
    expect(updater([msg("x"), msg("y"), msg("z"), msg("w")])).toHaveLength(4);
  });

  it("waits for fresh reference after streaming→ready, then force-hydrates", () => {
    const setMessages = vi.fn();
    const staleSnapshot = [msg("s1")];

    const { rerender } = renderHook(
      ({
        status,
        hydratedMessages,
      }: {
        status: "streaming" | "ready";
        hydratedMessages: UIMessage[];
      }) =>
        useHydrateOnStreamEnd({
          status,
          hydratedMessages,
          isReconnectScheduled: false,
          hasActiveStream: false,
          setMessages,
        }),
      {
        initialProps: {
          status: "streaming",
          hydratedMessages: staleSnapshot,
        },
      },
    );

    // While streaming, no hydration activity.
    expect(setMessages).not.toHaveBeenCalled();

    // Flip to ready with the SAME hydrated reference (stale) — the force-
    // hydrate flag is armed, but we must not overwrite yet.
    rerender({ status: "ready", hydratedMessages: staleSnapshot });
    expect(setMessages).not.toHaveBeenCalled();

    // New reference arrives → force-hydrate runs once.
    const fresh = [msg("s1"), msg("s2")];
    rerender({ status: "ready", hydratedMessages: fresh });
    expect(setMessages).toHaveBeenCalledTimes(1);
    const arg = setMessages.mock.calls[0][0];
    // Force-hydrate replaces unconditionally (not an updater fn).
    expect(Array.isArray(arg)).toBe(true);
    expect((arg as UIMessage[]).map((m) => m.id)).toEqual(["s1", "s2"]);

    // Subsequent rerender with same fresh ref → no additional call.
    rerender({ status: "ready", hydratedMessages: fresh });
    expect(setMessages).toHaveBeenCalledTimes(1);
  });

  it("ignores undefined or empty hydratedMessages", () => {
    const setMessages = vi.fn();
    renderHook(() =>
      useHydrateOnStreamEnd({
        status: "ready",
        hydratedMessages: undefined,
        isReconnectScheduled: false,
        hasActiveStream: false,
        setMessages,
      }),
    );
    expect(setMessages).not.toHaveBeenCalled();
  });

  it("resolves in-progress parts + toasts when hydrated state is zombied and no active stream", () => {
    mockToast.mockClear();
    const setMessages = vi.fn();
    const zombie: UIMessage = {
      id: "a",
      role: "assistant",
      parts: [
        {
          type: "text" as const,
          text: "half-written reply",
          state: "streaming" as const,
        },
      ],
    };

    renderHook(() =>
      useHydrateOnStreamEnd({
        status: "ready",
        hydratedMessages: [zombie],
        isReconnectScheduled: false,
        hasActiveStream: false,
        setMessages,
      }),
    );

    expect(mockToast).toHaveBeenCalledTimes(1);
    expect(mockToast.mock.calls[0][0]).toMatchObject({
      title: "Previous response was interrupted",
    });
    // setMessages is invoked once with a length-gated updater — pass a short
    // prev and expect the replaced (finalised) messages back.
    expect(setMessages).toHaveBeenCalledTimes(1);
    const updater = setMessages.mock.calls[0][0] as (
      prev: UIMessage[],
    ) => UIMessage[];
    const finalised = updater([]);
    const last = finalised[finalised.length - 1];
    const lastPart = last.parts[last.parts.length - 1];
    expect(lastPart.type).toBe("text");
    // Last appended part is the interrupted marker.
    expect((lastPart as { text: string }).text.includes("interrupted")).toBe(
      true,
    );
  });

  it("leaves zombie parts alone when backend still has active stream", () => {
    mockToast.mockClear();
    const setMessages = vi.fn();
    const partial: UIMessage = {
      id: "a",
      role: "assistant",
      parts: [
        {
          type: "text" as const,
          text: "still writing",
          state: "streaming" as const,
        },
      ],
    };

    renderHook(() =>
      useHydrateOnStreamEnd({
        status: "ready",
        hydratedMessages: [partial],
        isReconnectScheduled: false,
        hasActiveStream: true,
        setMessages,
      }),
    );

    expect(mockToast).not.toHaveBeenCalled();
  });
});
