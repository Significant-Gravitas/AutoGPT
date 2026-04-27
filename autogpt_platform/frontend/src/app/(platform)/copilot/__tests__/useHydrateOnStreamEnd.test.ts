import { renderHook } from "@testing-library/react";
import type { UIMessage } from "ai";
import { describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

import {
  _resetInterruptedToastLedgerForTests,
  useHydrateOnStreamEnd,
} from "../useHydrateOnStreamEnd";

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
        sessionId: null,
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
        sessionId: null,
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
        sessionId: null,
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
          sessionId: null,
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
    const updater = setMessages.mock.calls[0][0];
    // Force-hydrate uses an updater so it can preserve any in-flight
    // promoted-* user bubbles whose text isn't in the hydrated DB rows.
    expect(typeof updater).toBe("function");
    expect(
      (updater as (prev: UIMessage[]) => UIMessage[])([]).map((m) => m.id),
    ).toEqual(["s1", "s2"]);

    // Subsequent rerender with same fresh ref → no additional call.
    rerender({ status: "ready", hydratedMessages: fresh });
    expect(setMessages).toHaveBeenCalledTimes(1);
  });

  it("preserves promoted user bubbles whose text isn't in the hydrated DB rows", () => {
    // Simulates: Claude saw a queued message via mid-turn injection but the
    // persist rolled back, so the DB has no matching user row. We must NOT
    // drop the in-flight bubble during force-hydrate.
    const setMessages = vi.fn();
    const stale = [msg("a")];
    const { rerender } = renderHook(
      ({
        status,
        hydratedMessages,
      }: {
        status: "submitted" | "streaming" | "ready" | "error";
        hydratedMessages: UIMessage[];
      }) =>
        useHydrateOnStreamEnd({
          sessionId: null,
          status,
          hydratedMessages,
          isReconnectScheduled: false,
          hasActiveStream: false,
          setMessages,
        }),
      { initialProps: { status: "streaming", hydratedMessages: stale } },
    );

    // Arm force-hydrate, then drop in a fresh hydrated reference.
    rerender({ status: "ready", hydratedMessages: stale });
    const fresh = [msg("a"), msg("b")];
    rerender({ status: "ready", hydratedMessages: fresh });
    expect(setMessages).toHaveBeenCalledTimes(1);

    const updater = setMessages.mock.calls[0][0] as (
      prev: UIMessage[],
    ) => UIMessage[];

    const promoted: UIMessage = {
      id: "promoted-midturn-xyz",
      role: "user",
      parts: [{ type: "text", text: "queued message", state: "done" }],
    };
    const result = updater([msg("a"), promoted, msg("b")]);
    // Promoted bubble survives because no hydrated user message contains its text.
    expect(result.map((m) => m.id)).toEqual(["a", "b", "promoted-midturn-xyz"]);
  });

  it("drops promoted bubbles whose text is already represented in the DB", () => {
    const setMessages = vi.fn();
    const stale = [msg("a")];
    const { rerender } = renderHook(
      ({
        status,
        hydratedMessages,
      }: {
        status: "submitted" | "streaming" | "ready" | "error";
        hydratedMessages: UIMessage[];
      }) =>
        useHydrateOnStreamEnd({
          sessionId: null,
          status,
          hydratedMessages,
          isReconnectScheduled: false,
          hasActiveStream: false,
          setMessages,
        }),
      { initialProps: { status: "streaming", hydratedMessages: stale } },
    );

    rerender({ status: "ready", hydratedMessages: stale });
    const persistedUser: UIMessage = {
      id: "session-seq-3",
      role: "user",
      parts: [{ type: "text", text: "queued message", state: "done" }],
    };
    const fresh = [msg("a"), persistedUser, msg("b")];
    rerender({ status: "ready", hydratedMessages: fresh });

    const updater = setMessages.mock.calls[0][0] as (
      prev: UIMessage[],
    ) => UIMessage[];

    const promoted: UIMessage = {
      id: "promoted-midturn-xyz",
      role: "user",
      parts: [{ type: "text", text: "queued message", state: "done" }],
    };
    const result = updater([msg("a"), promoted, msg("b")]);
    // No duplicate — DB user row already covers the bubble's text.
    expect(result.map((m) => m.id)).toEqual(["a", "session-seq-3", "b"]);
  });

  it("ignores undefined or empty hydratedMessages", () => {
    const setMessages = vi.fn();
    renderHook(() =>
      useHydrateOnStreamEnd({
        sessionId: null,
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
        sessionId: "zombie-session",
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
        sessionId: "active-session",
        status: "ready",
        hydratedMessages: [partial],
        isReconnectScheduled: false,
        hasActiveStream: true,
        setMessages,
      }),
    );

    expect(mockToast).not.toHaveBeenCalled();
  });

  it("does not fire interrupted toast against the pre-turn stale snapshot", () => {
    mockToast.mockClear();
    _resetInterruptedToastLedgerForTests();

    const setMessages = vi.fn();
    const stale: UIMessage[] = [
      {
        id: "a",
        role: "assistant",
        parts: [
          {
            type: "text" as const,
            text: "half-written reply",
            state: "streaming" as const,
          },
        ],
      },
    ];

    const { rerender } = renderHook(
      ({
        status,
        hydratedMessages,
      }: {
        status: "streaming" | "ready";
        hydratedMessages: UIMessage[];
      }) =>
        useHydrateOnStreamEnd({
          sessionId: "sess-stale-toast",
          status,
          hydratedMessages,
          isReconnectScheduled: false,
          hasActiveStream: false,
          setMessages,
        }),
      { initialProps: { status: "streaming", hydratedMessages: stale } },
    );

    // status flip with the SAME stale snapshot — force-hydrate is armed,
    // but the data is still the pre-turn ref. Toast must NOT fire here:
    // the interrupted bubble may not exist once the refetch lands.
    rerender({ status: "ready", hydratedMessages: stale });
    expect(mockToast).not.toHaveBeenCalled();
    expect(setMessages).not.toHaveBeenCalled();

    // Fresh ref arrives, still has zombie parts → toast fires alongside
    // the apply.
    const fresh: UIMessage[] = [
      {
        id: "a",
        role: "assistant",
        parts: [
          {
            type: "text" as const,
            text: "half-written reply",
            state: "streaming" as const,
          },
        ],
      },
    ];
    rerender({ status: "ready", hydratedMessages: fresh });
    expect(mockToast).toHaveBeenCalledTimes(1);
  });

  it("only fires the interrupted toast once per session across remounts", () => {
    mockToast.mockClear();
    _resetInterruptedToastLedgerForTests();

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

    // Mount #1 — toast fires.
    const first = renderHook(() =>
      useHydrateOnStreamEnd({
        sessionId: "sess-toast",
        status: "ready",
        hydratedMessages: [zombie],
        isReconnectScheduled: false,
        hasActiveStream: false,
        setMessages,
      }),
    );
    expect(mockToast).toHaveBeenCalledTimes(1);
    first.unmount();

    // Mount #2 — same sessionId, fresh ref-state but already-shown ledger.
    // Reproduces "switch away then back" — useRef would reset and re-toast.
    renderHook(() =>
      useHydrateOnStreamEnd({
        sessionId: "sess-toast",
        status: "ready",
        hydratedMessages: [zombie],
        isReconnectScheduled: false,
        hasActiveStream: false,
        setMessages,
      }),
    );
    expect(mockToast).toHaveBeenCalledTimes(1);

    // Different session id → new toast is allowed.
    renderHook(() =>
      useHydrateOnStreamEnd({
        sessionId: "sess-toast-other",
        status: "ready",
        hydratedMessages: [zombie],
        isReconnectScheduled: false,
        hasActiveStream: false,
        setMessages,
      }),
    );
    expect(mockToast).toHaveBeenCalledTimes(2);
  });
});
