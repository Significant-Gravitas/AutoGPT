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
