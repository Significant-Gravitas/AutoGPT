import { cleanup, renderHook } from "@testing-library/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  _resetInterruptedToastLedgerForTests,
  useHydrateOnStreamEnd,
} from "../useHydrateOnStreamEnd";
import { CANCELLED_MARKER } from "../useCopilotStop";

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
  statuses = ["ready"],
}: {
  prev: Messages;
  staleWindow: Messages;
  freshWindow: Messages;
  /** What the chat passes through after streaming, before the refetch lands. */
  statuses?: Parameters<typeof useHydrateOnStreamEnd>[0]["status"][];
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
  for (const status of statuses) {
    rerender({
      ...baseProps,
      status,
      hydratedMessages: staleWindow,
    } satisfies Props);
  }

  // React Query swaps in the fresh post-turn window → force-hydrate applies.
  rerender({
    ...baseProps,
    status: statuses[statuses.length - 1],
    hydratedMessages: freshWindow,
  } satisfies Props);

  return captured;
}

describe("useHydrateOnStreamEnd — sliding-window history retention (SECRT-2424)", () => {
  it.each(["Hello [1] world", "Goodbye", ""])(
    "reconciles the streamed preview with rewritten provider text: %j",
    (finalText) => {
      const preview: Messages = [
        seqMessage(1),
        {
          id: "live-assistant",
          role: "assistant",
          parts: [{ type: "text", text: "Hello world", state: "done" }],
        },
      ];
      const canonical: Messages = [
        seqMessage(1),
        {
          id: `${SESSION_ID}-seq-2`,
          role: "assistant",
          parts: [{ type: "text", text: finalText, state: "done" }],
        },
      ];
      expect(
        runForceHydrate({
          prev: preview,
          staleWindow: [seqMessage(1)],
          freshWindow: canonical,
        }),
      ).toEqual(canonical);
    },
  );

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

describe("useHydrateOnStreamEnd — a turn the user stopped", () => {
  afterEach(() => {
    _resetInterruptedToastLedgerForTests();
    cleanup();
  });

  function stoppedAssistant(): UIMessage<unknown, UIDataTypes, UITools> {
    return {
      id: "live-assistant",
      role: "assistant",
      parts: [
        { type: "text", text: "Half an answer", state: "done" },
        { type: "text", text: CANCELLED_MARKER, state: "done" },
      ],
    };
  }

  it("keeps the partial and its marker when the refetch has no reply for the turn", () => {
    const result = runForceHydrate({
      prev: [seqMessage(1), stoppedAssistant()],
      staleWindow: [],
      freshWindow: [seqMessage(1)],
    });

    expect(result).toEqual([seqMessage(1), stoppedAssistant()]);
  });

  it("defers to the database once it carries the stopped turn", () => {
    const persisted: Messages = [
      seqMessage(1),
      {
        id: `${SESSION_ID}-seq-2`,
        role: "assistant",
        parts: [
          {
            type: "text",
            text: `Half an answer ${CANCELLED_MARKER}`,
            state: "done",
          },
        ],
      },
    ];

    expect(
      runForceHydrate({
        prev: [seqMessage(1), stoppedAssistant()],
        staleWindow: [],
        freshWindow: persisted,
      }),
    ).toEqual(persisted);
  });
});

describe("useHydrateOnStreamEnd — a send the backend has not written yet", () => {
  afterEach(() => {
    _resetInterruptedToastLedgerForTests();
    cleanup();
  });

  function row(
    seq: number,
    role: "user" | "assistant",
    text: string,
  ): UIMessage<unknown, UIDataTypes, UITools> {
    return {
      id: `${SESSION_ID}-seq-${seq}`,
      role,
      parts: [{ type: "text", text, state: "done" }],
    };
  }

  it('keeps a second "yes" that the earlier "yes" row does not account for', () => {
    const answered = [row(1, "user", "yes"), row(2, "assistant", "Booked.")];
    const queued: UIMessage<unknown, UIDataTypes, UITools> = {
      id: "local-send",
      role: "user",
      parts: [{ type: "text", text: "yes" }],
    };

    const result = runForceHydrate({
      prev: [...answered, queued],
      staleWindow: [row(1, "user", "yes")],
      freshWindow: answered,
    });

    expect(result).toEqual([...answered, queued]);
  });

  function local(
    id: string,
    role: "user" | "assistant",
    text: string,
  ): UIMessage<unknown, UIDataTypes, UITools> {
    return { id, role, parts: [{ type: "text", text }] };
  }

  it.each([
    {
      // A long turn pushed the prompt's row out of the latest-rows window.
      name: "a prompt whose row lies before the refetched window",
      prev: [
        row(1, "user", "Plan it"),
        row(2, "assistant", "Planned."),
        local("local-prompt", "user", "Build it"),
        local("live-answer", "assistant", "Built."),
      ],
      fresh: range(60, 70),
      expected: [
        row(1, "user", "Plan it"),
        row(2, "assistant", "Planned."),
        ...range(60, 70),
      ],
    },
    {
      name: "a first send refused before it was saved, then retried",
      prev: [
        local("refused", "user", "yes"),
        local("retried", "user", "yes"),
        local("live-answer", "assistant", "Booked."),
      ],
      // A session's first row is sequence 0.
      fresh: [row(0, "user", "yes"), row(1, "assistant", "Booked.")],
      expected: [row(0, "user", "yes"), row(1, "assistant", "Booked.")],
    },
    {
      name: "a send whose stream failed",
      prev: [row(0, "user", "Plan it"), local("refused", "user", "Build it")],
      fresh: [row(0, "user", "Plan it")],
      expected: [row(0, "user", "Plan it")],
      statuses: ["error" as const],
    },
    {
      name: "a refused send once its retry has ended cleanly",
      prev: [
        row(0, "user", "Plan it"),
        row(1, "assistant", "Planned."),
        local("refused", "user", "yes"),
        local("retried", "user", "yes"),
        local("live-answer", "assistant", "Booked."),
      ],
      fresh: [
        row(0, "user", "Plan it"),
        row(1, "assistant", "Planned."),
        row(2, "user", "yes"),
        row(3, "assistant", "Booked."),
      ],
      expected: [
        row(0, "user", "Plan it"),
        row(1, "assistant", "Planned."),
        row(2, "user", "yes"),
        row(3, "assistant", "Booked."),
      ],
      statuses: [
        "error" as const,
        "submitted" as const,
        "streaming" as const,
        "ready" as const,
      ],
    },
  ])("does not keep $name", ({ prev, fresh, expected, statuses }) => {
    expect(
      runForceHydrate({ prev, staleWindow: [], freshWindow: fresh, statuses }),
    ).toEqual(expected);
  });

  it("leaves a promoted bubble's row to it, not to a later bubble with the same text", () => {
    const promoted = local("promoted-midturn-1", "user", "yes");
    const queued = local("local-send", "user", "yes");
    const stored = [
      row(1, "user", "Book both"),
      row(2, "user", "yes"),
      row(3, "assistant", "Booked."),
    ];

    const result = runForceHydrate({
      prev: [row(1, "user", "Book both"), promoted, queued],
      staleWindow: [],
      freshWindow: stored,
    });

    expect(result).toEqual([...stored, queued]);
  });

  it("gives a promoted bubble its own later row, not an earlier one with the same text", () => {
    const answered = [row(0, "user", "yes"), row(1, "assistant", "Booked.")];
    const promoted = local("promoted-midturn-1", "user", "yes");
    const queued = local("local-send", "user", "yes");
    const stored = [...answered, row(2, "user", "yes")];

    const result = runForceHydrate({
      prev: [...answered, promoted, queued],
      staleWindow: [],
      freshWindow: stored,
    });

    expect(result).toEqual([...stored, queued]);
  });
});
