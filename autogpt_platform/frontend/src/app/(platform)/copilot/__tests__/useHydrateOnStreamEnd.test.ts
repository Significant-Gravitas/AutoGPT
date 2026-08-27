import { cleanup, renderHook } from "@testing-library/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  _resetInterruptedToastLedgerForTests,
  useHydrateOnStreamEnd,
} from "../useHydrateOnStreamEnd";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
}));

type Messages = UIMessage<unknown, UIDataTypes, UITools>[];

const SESSION_ID = "session-1";

function seqMessage(seq: number): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id: `${SESSION_ID}-seq-${seq}`,
    role: seq % 2 === 0 ? "assistant" : "user",
    parts: [{ type: "text", text: `message ${seq}`, state: "done" }],
  };
}

function range(start: number, end: number): Messages {
  const out: Messages = [];
  for (let seq = start; seq <= end; seq++) out.push(seqMessage(seq));
  return out;
}

function seqOf(message: UIMessage): number {
  const match = /-seq-(\d+)$/.exec(message.id);
  return match ? Number(match[1]) : NaN;
}

/**
 * Drive the hook through a stream-end force-hydrate and capture the result
 * `setMessages` produces. `prev` is the in-memory AI-SDK state at the moment
 * the force-hydrate applies; `staleWindow`/`freshWindow` are the two distinct
 * `hydratedMessages` references React Query swaps in (stale = pre-turn, fresh
 * = the post-turn refetch).
 */
function runForceHydrate({
  prev,
  staleWindow,
  freshWindow,
}: {
  prev: Messages;
  staleWindow: Messages;
  freshWindow: Messages;
}): Messages | null {
  let captured: Messages | null = null;
  const setMessages = vi.fn(
    (updater: Messages | ((p: Messages) => Messages)) => {
      captured = typeof updater === "function" ? updater(prev) : updater;
    },
  );

  type Props = Parameters<typeof useHydrateOnStreamEnd>[0];
  const baseProps = {
    sessionId: SESSION_ID,
    isReconnectScheduled: false,
    hasActiveStream: false,
    isFinishProbing: false,
    setMessages,
  };

  const { rerender } = renderHook<void, Props>(
    (props) => useHydrateOnStreamEnd(props),
    {
      initialProps: {
        ...baseProps,
        status: "streaming",
        hydratedMessages: staleWindow,
      },
    },
  );

  // Stream ends → arms force-hydrate, snapshots the stale (pre-turn) window.
  rerender({
    ...baseProps,
    status: "ready",
    hydratedMessages: staleWindow,
  } satisfies Props);

  // React Query swaps in the fresh post-turn window → force-hydrate applies.
  rerender({
    ...baseProps,
    status: "ready",
    hydratedMessages: freshWindow,
  } satisfies Props);

  return captured;
}

describe("useHydrateOnStreamEnd — sliding-window history retention (SECRT-2424)", () => {
  afterEach(() => {
    _resetInterruptedToastLedgerForTests();
    cleanup();
  });

  it("keeps middle messages when the tail window slides past loaded history", () => {
    // Before the turn, memory holds seq 51-102 (the prior window plus the
    // turn that just streamed). The refetched window only covers seq 53-102,
    // so a blind replace would drop seq 51 and 52 into a hole between the
    // recent window and any older `pagedMessages`.
    const prev = range(51, 102);
    const freshWindow = range(53, 102);

    const result = runForceHydrate({
      prev,
      staleWindow: range(51, 100),
      freshWindow,
    });

    expect(result).not.toBeNull();
    const seqs = result!.map(seqOf);
    // No hole: 51 and 52 survive, contiguous through 102.
    expect(seqs).toEqual(range(51, 102).map(seqOf));
    expect(seqs).toContain(51);
    expect(seqs).toContain(52);
  });

  it("retains all older messages when the window omits the start of a grown session", () => {
    // Session grew from <50 to 60 messages with no scroll-back: the window
    // is seq 11-60, but memory still holds seq 1-60. seq 1-10 must survive.
    const result = runForceHydrate({
      prev: range(1, 60),
      staleWindow: range(1, 50),
      freshWindow: range(11, 60),
    });

    expect(result).not.toBeNull();
    expect(result!.map(seqOf)).toEqual(range(1, 60).map(seqOf));
  });

  it("replaces with the window unchanged when nothing older is retained", () => {
    // Memory and the fresh window cover the same range — no older tail to
    // keep, so the canonical window wins outright (no duplicates).
    const freshWindow = range(53, 102);
    const result = runForceHydrate({
      prev: range(53, 102),
      staleWindow: range(53, 100),
      freshWindow,
    });

    expect(result).not.toBeNull();
    expect(result!.map(seqOf)).toEqual(freshWindow.map(seqOf));
  });

  it("retains all of prev when the window shares no sequence with memory", () => {
    // Pathological no-overlap: every in-memory message is older than the whole
    // refetched window (the leading run is the entire array). All of prev must
    // survive, prepended to the disjoint window.
    const prev = range(1, 40);
    const freshWindow = range(60, 102);
    const result = runForceHydrate({
      prev,
      staleWindow: range(60, 100),
      freshWindow,
    });

    expect(result).not.toBeNull();
    expect(result!.map(seqOf)).toEqual([
      ...range(1, 40).map(seqOf),
      ...freshWindow.map(seqOf),
    ]);
  });
});

describe("useHydrateOnStreamEnd — continuation-turn flash guard", () => {
  type Props = Parameters<typeof useHydrateOnStreamEnd>[0];

  afterEach(() => {
    _resetInterruptedToastLedgerForTests();
    cleanup();
  });

  function makeCapturingSetMessages(prev: Messages) {
    const captured: { current: Messages | null } = { current: null };
    const setMessages = vi.fn(
      (updater: Messages | ((p: Messages) => Messages)) => {
        captured.current =
          typeof updater === "function" ? updater(prev) : updater;
      },
    );
    return { setMessages, captured };
  }

  function setup({ prev = [] as Messages } = {}) {
    const { setMessages, captured } = makeCapturingSetMessages(prev);
    const baseProps = {
      sessionId: SESSION_ID,
      isReconnectScheduled: false,
      hasActiveStream: false,
      isFinishProbing: false,
      setMessages,
    };
    const hook = renderHook<void, Props>(
      (props) => useHydrateOnStreamEnd(props),
      {
        initialProps: {
          ...baseProps,
          status: "streaming",
          hydratedMessages: range(51, 100),
        },
      },
    );
    // Stream ends → arms the force-hydrate against the stale pre-turn window.
    hook.rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(51, 100),
    } satisfies Props);
    return { ...hook, baseProps, setMessages, captured };
  }

  it("defers the force-hydrate while the post-finish probe is in flight", () => {
    const { rerender, baseProps, setMessages, captured } = setup();

    // Fresh post-turn window lands while `handleFinish` is still probing for
    // a continuation turn — the replace must NOT apply (it would swap every
    // message id, remount the list and flash the panel).
    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      isFinishProbing: true,
    } satisfies Props);
    expect(setMessages).not.toHaveBeenCalled();

    // Probe settles with no continuation → the replace lands.
    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      isFinishProbing: false,
    } satisfies Props);
    expect(setMessages).toHaveBeenCalledTimes(1);
    expect(captured.current!.map(seqOf)).toEqual(range(53, 102).map(seqOf));
  });

  it("holds the force-hydrate while the backend still has an active stream", () => {
    const { rerender, baseProps, setMessages, captured } = setup();

    // Refetched session data shows a continuation turn already live — the
    // resume flow takes over, so the replace must wait.
    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      hasActiveStream: true,
    } satisfies Props);
    expect(setMessages).not.toHaveBeenCalled();

    // Backend goes idle → the replace lands.
    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      hasActiveStream: false,
    } satisfies Props);
    expect(setMessages).toHaveBeenCalledTimes(1);
    expect(captured.current!.map(seqOf)).toEqual(range(53, 102).map(seqOf));
  });

  it("stays held through the probe → reconnect handoff", () => {
    const { rerender, baseProps, setMessages } = setup();

    // The real continuation flow: the probe is in flight...
    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      isFinishProbing: true,
    } satisfies Props);
    // ...then the probe finds an active stream and schedules the reconnect —
    // `isFinishProbing` flips false while `hasActiveStream` is now true. The
    // replace must stay held by the second gate.
    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      isFinishProbing: false,
      hasActiveStream: true,
    } satisfies Props);
    expect(setMessages).not.toHaveBeenCalled();

    // Continuation turn ends, backend idle → the replace finally lands.
    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      hasActiveStream: false,
    } satisfies Props);
    expect(setMessages).toHaveBeenCalledTimes(1);
  });

  it("retains older in-memory history when the deferred replace lands", () => {
    // Memory holds seq 1-60; the fresh window covers seq 53-102. The delayed
    // replace must still prepend the older in-memory run (seq 1-52) instead
    // of tearing a hole into scrolled-back history.
    const { rerender, baseProps, setMessages, captured } = setup({
      prev: range(1, 60),
    });

    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      isFinishProbing: true,
    } satisfies Props);
    expect(setMessages).not.toHaveBeenCalled();

    rerender({
      ...baseProps,
      status: "ready",
      hydratedMessages: range(53, 102),
      isFinishProbing: false,
    } satisfies Props);
    expect(setMessages).toHaveBeenCalledTimes(1);
    expect(captured.current!.map(seqOf)).toEqual(range(1, 102).map(seqOf));
  });

  it("still applies length-gated top-ups while a stream is active (session restore)", () => {
    // Reopening a session with an active backend stream relies on the top-up
    // branch to render history before the resume replay starts — only the
    // force-hydrate path is gated on `hasActiveStream`.
    const { setMessages, captured } = makeCapturingSetMessages([]);
    renderHook(() =>
      useHydrateOnStreamEnd({
        sessionId: SESSION_ID,
        status: "ready",
        hydratedMessages: range(1, 10),
        isReconnectScheduled: false,
        hasActiveStream: true,
        isFinishProbing: false,
        setMessages,
      }),
    );
    expect(setMessages).toHaveBeenCalledTimes(1);
    expect(captured.current!.map(seqOf)).toEqual(range(1, 10).map(seqOf));
  });
});
