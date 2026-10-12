import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getV2GetPendingMessages } from "@/app/api/__generated__/endpoints/chat/chat";
import { splitMessagesAtDrainHints } from "./components/ChatMessagesContainer/midTurnSplit";
import { useCopilotPendingChips } from "./useCopilotPendingChips";

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  getV2GetPendingMessages: vi.fn(),
}));

type Messages = UIMessage<unknown, UIDataTypes, UITools>[];
type ChatStatus = Parameters<typeof useCopilotPendingChips>[0]["status"];

const mockGetPending = vi.mocked(getV2GetPendingMessages);

const ASSISTANT_ID = "assistant-stable-id";

function assistantMessage(
  drainHints = 0,
  /** Text the backend ships on the hint. Present from the release that
   *  renders the follow-up bubble at the drain point. */
  drainedText?: string,
): Messages[number] {
  const parts: Messages[number]["parts"] = [
    { type: "text", text: "working…", state: "done" },
  ];
  for (let i = 0; i < drainHints; i++) {
    parts.push({
      type: "data-pending-drained",
      id: `hint-${i}`,
      data: {
        drainedCount: 1,
        ...(drainedText
          ? { messages: [{ id: `pm-${i}`, content: drainedText }] }
          : {}),
      },
    } as Messages[number]["parts"][number]);
  }
  return { id: ASSISTANT_ID, role: "assistant", parts };
}

type DrainHint = { text: string } | "count-only";

/** One assistant message carrying the given hints in order, with a tool
 *  result drawn between them so each one is a split point. */
function assistantWithHints(hints: DrainHint[]): Messages[number] {
  const parts: Messages[number]["parts"] = [
    { type: "text", text: "working…", state: "done" },
  ];
  hints.forEach((hint, i) => {
    parts.push({
      type: "data-pending-drained",
      id: `hint-${i}`,
      data:
        hint === "count-only"
          ? { drainedCount: 1 }
          : {
              drainedCount: 1,
              messages: [{ id: `pm-${i}`, content: hint.text }],
            },
    } as Messages[number]["parts"][number]);
    parts.push({ type: "text", text: `step ${i}`, state: "done" });
  });
  return { id: ASSISTANT_ID, role: "assistant", parts };
}

/** The auto-continue assistant, optionally carrying drain hints of its
 *  own with a visible step between them so each is a split point. */
function continuationMessage(hints: DrainHint[] = []): Messages[number] {
  return { ...assistantWithHints(hints), id: "assistant-continuation" };
}

/** Hold every pending-buffer GET open until the test resolves it, so two
 *  reconciliations of the same chip can overlap the way they do live. */
function deferPendingGets() {
  const resolvers: Array<(count: number) => void> = [];
  mockGetPending.mockImplementation(
    () =>
      new Promise((resolve) => {
        resolvers.push((count) =>
          resolve({
            status: 200,
            data: { count, messages: [] },
            headers: new Headers(),
          } as Awaited<ReturnType<typeof getV2GetPendingMessages>>),
        );
      }),
  );
  return {
    count: () => resolvers.length,
    resolveDrained: (index: number) =>
      act(async () => {
        resolvers[index](0);
      }),
  };
}

function storedFallbacks(messages: Messages) {
  return messages.filter((m) => m.id.startsWith("promoted-"));
}

/** What the transcript draws: the user rows of the render-time split. */
function renderedUserRows(messages: Messages) {
  return splitMessagesAtDrainHints(messages)
    .filter((row) => row.role === "user")
    .map((row) => ({
      id: row.id,
      text: row.parts
        .map((part) => (part.type === "text" ? part.text : ""))
        .join(""),
    }));
}

// Server reports the buffer as fully drained (count 0) so the hook promotes
// every locally-queued chip to a bubble.
function mockBufferDrained() {
  mockGetPending.mockResolvedValue({
    status: 200,
    data: { count: 0, messages: [] },
    headers: new Headers(),
  } as Awaited<ReturnType<typeof getV2GetPendingMessages>>);
}

function setupHook(initialMessages: Messages) {
  let current: Messages = initialMessages;
  const setMessages = vi.fn(
    (updater: Messages | ((prev: Messages) => Messages)) => {
      current = typeof updater === "function" ? updater(current) : updater;
    },
  );

  const view = renderHook(
    ({ messages }) =>
      useCopilotPendingChips({
        sessionId: "s1",
        status: "streaming",
        messages,
        setMessages,
      }),
    { initialProps: { messages: initialMessages } },
  );

  /** Rerender with a new transcript, as `useChat` handing the hook a new
   *  `messages` array would, so `getMessages` keeps tracking it. */
  function rerender(messages: Messages) {
    current = messages;
    view.rerender({ messages });
  }

  return { view, setMessages, getMessages: () => current, rerender };
}

describe("useCopilotPendingChips", () => {
  beforeEach(() => {
    mockGetPending.mockReset();
    mockBufferDrained();
  });
  afterEach(cleanup);

  it("queues a message on a plain-HTTP LAN origin, where crypto.randomUUID is missing", () => {
    const originalCrypto = globalThis.crypto;
    vi.stubGlobal("crypto", {
      getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto),
    });
    try {
      const { view } = setupHook([assistantMessage(0)]);

      act(() => {
        view.result.current.queueMessage("sent while a turn was streaming");
      });

      expect(view.result.current.queuedMessages).toEqual([
        "sent while a turn was streaming",
      ]);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("promotes a queued chip to a bubble the instant a data-pending-drained hint arrives", async () => {
    const { view, getMessages } = setupHook([assistantMessage(0)]);

    act(() => {
      view.result.current.queueMessage("follow up");
    });
    expect(view.result.current.queuedMessages).toEqual(["follow up"]);

    // The backend drains mid-turn and pushes the SSE hint: rerender with the
    // new data-pending-drained part on the streaming assistant message.
    await act(async () => {
      view.rerender({ messages: [assistantMessage(1)] });
    });

    await waitFor(() => {
      expect(mockGetPending).toHaveBeenCalledWith("s1");
      // Chip cleared from the strip…
      expect(view.result.current.queuedMessages).toEqual([]);
    });

    // …and promoted to a user bubble inserted before the streaming assistant.
    const promoted = getMessages().find((m) =>
      m.id.startsWith("promoted-midturn-pending-chip-"),
    );
    expect(promoted?.role).toBe("user");
  });

  it("draws the follow-up once when the hint carries the text", async () => {
    // The transcript renders the bubble at the drain point itself
    // (splitMessagesAtDrainHints); the fallback bubble promoted here must
    // not show up next to it.
    const { view, getMessages, rerender } = setupHook([assistantMessage(0)]);

    act(() => {
      view.result.current.queueMessage("follow up");
    });

    await act(async () => {
      rerender([assistantMessage(1, "follow up")]);
    });

    await waitFor(() => {
      expect(mockGetPending).toHaveBeenCalledWith("s1");
      expect(view.result.current.queuedMessages).toEqual([]);
    });
    expect(renderedUserRows(getMessages())).toEqual([
      { id: "midturn-pm-0", text: "follow up" },
    ]);
  });

  it("draws one bubble when the backstop poll beat the text-bearing hint", async () => {
    vi.useFakeTimers();
    try {
      const { view, getMessages, rerender } = setupHook([assistantMessage(0)]);
      act(() => {
        view.result.current.queueMessage("follow up");
      });

      // The GET observes the drained buffer before the SSE hint reaches
      // the client: the chip becomes a fallback bubble above the assistant.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });
      expect(view.result.current.queuedMessages).toEqual([]);
      const fallback = getMessages().find((m) =>
        m.id.startsWith("promoted-midturn-pending-chip-"),
      );
      expect(fallback?.role).toBe("user");
      expect(renderedUserRows(getMessages())).toEqual([
        { id: fallback!.id, text: "follow up" },
      ]);

      // Now the hint lands with the text. Exactly one follow-up must render,
      // between the two assistant segments, and the live assistant must
      // still be the last message so later deltas keep landing in it.
      await act(async () => {
        rerender([
          ...getMessages().filter((m) => m.role !== "assistant"),
          assistantWithHints([{ text: "follow up" }]),
        ]);
      });
      const rows = splitMessagesAtDrainHints(getMessages());
      expect(rows.map((row) => row.id)).toEqual([
        `${ASSISTANT_ID}#seg0`,
        "midturn-pm-0",
        ASSISTANT_ID,
      ]);
      const messages = getMessages();
      expect(messages[messages.length - 1].id).toBe(ASSISTANT_ID);
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps two identical follow-ups when only the first drain carried text", async () => {
    // Turn-scoped reconciliation: the first "continue" was drawn by its
    // hint, so its chip left no bubble behind. The second "continue" drains
    // through a count-only hint (older backend, rolling deploy) and must
    // still become a bubble — the earlier hint is not its drain.
    const { view, getMessages, rerender } = setupHook([assistantMessage(0)]);

    act(() => {
      view.result.current.queueMessage("continue");
    });
    await act(async () => {
      rerender([assistantWithHints([{ text: "continue" }])]);
    });
    await waitFor(() => {
      expect(view.result.current.queuedMessages).toEqual([]);
    });
    expect(renderedUserRows(getMessages()).map((row) => row.text)).toEqual([
      "continue",
    ]);

    act(() => {
      view.result.current.queueMessage("continue");
    });
    await act(async () => {
      rerender([
        ...getMessages().filter((m) => m.role !== "assistant"),
        assistantWithHints([{ text: "continue" }, "count-only"]),
      ]);
    });
    await waitFor(() => {
      expect(view.result.current.queuedMessages).toEqual([]);
    });

    expect(renderedUserRows(getMessages()).map((row) => row.text)).toEqual([
      "continue",
      "continue",
    ]);
  });

  it("keeps two identical follow-ups when the second drain's hint was dropped", async () => {
    vi.useFakeTimers();
    try {
      const { view, getMessages, rerender } = setupHook([assistantMessage(0)]);

      act(() => {
        view.result.current.queueMessage("continue");
      });
      await act(async () => {
        rerender([assistantWithHints([{ text: "continue" }])]);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(view.result.current.queuedMessages).toEqual([]);

      // No second hint ever arrives; the backstop poll sees the drained
      // buffer and must not let the first drain's text stand in for it.
      act(() => {
        view.result.current.queueMessage("continue");
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });
      expect(view.result.current.queuedMessages).toEqual([]);

      expect(renderedUserRows(getMessages()).map((row) => row.text)).toEqual([
        "continue",
        "continue",
      ]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("promotes only the copy of a same-batch repeated text the stream did not draw", async () => {
    const { view, getMessages, rerender } = setupHook([assistantMessage(0)]);

    act(() => {
      view.result.current.queueMessage("again");
      view.result.current.queueMessage("again");
    });
    // Both chips drain in one batch, but the hint only names one of them.
    await act(async () => {
      rerender([assistantWithHints([{ text: "again" }])]);
    });
    await waitFor(() => {
      expect(view.result.current.queuedMessages).toEqual([]);
    });

    expect(renderedUserRows(getMessages()).map((row) => row.text)).toEqual([
      "again",
      "again",
    ]);
  });

  it("still promotes a count-only drain that follows a text-bearing one", async () => {
    // Mixed hints reach one transcript across a rolling backend deploy. The
    // earlier hint's text is drawn by the split, but the later count-only
    // drain has no text on it, so its chip must still become a bubble —
    // the queue entry is dropped either way.
    function mixedHints(): Messages[number] {
      const parts: Messages[number]["parts"] = [
        { type: "text", text: "working…", state: "done" },
        {
          type: "data-pending-drained",
          id: "hint-text",
          data: {
            drainedCount: 1,
            messages: [{ id: "pm-0", content: "earlier follow up" }],
          },
        } as Messages[number]["parts"][number],
        {
          type: "data-pending-drained",
          id: "hint-count-only",
          data: { drainedCount: 1 },
        } as Messages[number]["parts"][number],
      ];
      return { id: ASSISTANT_ID, role: "assistant", parts };
    }

    const { view, getMessages, rerender } = setupHook([assistantMessage(0)]);

    act(() => {
      view.result.current.queueMessage("later follow up");
    });

    await act(async () => {
      rerender([mixedHints()]);
    });

    await waitFor(() => {
      expect(mockGetPending).toHaveBeenCalledWith("s1");
      expect(view.result.current.queuedMessages).toEqual([]);
    });

    const promoted = getMessages().find((m) =>
      m.id.startsWith("promoted-midturn-pending-chip-"),
    );
    expect(promoted?.role).toBe("user");
  });

  it("keeps the chip queued when useChat swaps its placeholder id while the backend still holds the message", async () => {
    mockGetPending.mockResolvedValue({
      status: 200,
      data: { count: 1, messages: ["follow up"] },
      headers: new Headers(),
    } as Awaited<ReturnType<typeof getV2GetPendingMessages>>);

    // The backend emits `data-status` before `start`, so the turn opens as
    // a status-only placeholder under the SDK's own id…
    const placeholder: Messages[number] = {
      id: "sdk-placeholder",
      role: "assistant",
      parts: [
        { type: "data-status", data: { message: "Preparing…" } },
      ] as Messages[number]["parts"],
    };
    const { view, getMessages, rerender } = setupHook([placeholder]);
    act(() => {
      view.result.current.queueMessage("follow up");
    });

    // …and the server's message id lands after it. That is not an
    // auto-continue: the follow-up is still in the buffer.
    await act(async () => {
      rerender([placeholder, assistantMessage(0)]);
    });

    await waitFor(() => expect(mockGetPending).toHaveBeenCalledWith("s1"));
    expect(view.result.current.queuedMessages).toEqual(["follow up"]);
    expect(getMessages().some((m) => m.id.startsWith("promoted-"))).toBe(false);

    // Every later delta lands in the same message: one reconciliation per
    // new id, not one per chunk.
    await act(async () => {
      rerender([placeholder, assistantMessage(0)]);
    });
    expect(mockGetPending).toHaveBeenCalledTimes(1);
  });

  it("promotes chips before the auto-continue assistant once the backend confirms the drain", async () => {
    const { view, getMessages, rerender } = setupHook([assistantMessage(0)]);
    act(() => {
      view.result.current.queueMessage("follow up");
    });

    const continuation: Messages[number] = {
      id: "assistant-continuation",
      role: "assistant",
      parts: [{ type: "text", text: "continuing…", state: "done" }],
    };
    await act(async () => {
      rerender([assistantMessage(0), continuation]);
    });

    await waitFor(() => {
      expect(mockGetPending).toHaveBeenCalledWith("s1");
      expect(view.result.current.queuedMessages).toEqual([]);
    });
    const ids = getMessages().map((m) => m.id);
    expect(ids).toHaveLength(3);
    expect(ids[0]).toBe(ASSISTANT_ID);
    expect(ids[1]).toMatch(/^promoted-auto-continue-pending-chip-/);
    expect(ids[2]).toBe("assistant-continuation");
  });

  describe("overlapping reconciliations of one queued chip", () => {
    // A new assistant id and a drain hint (or the backstop poll) can each
    // start a GET for the same chip before the other resolves. Each sees a
    // drained buffer; only one bubble may be stored for the chip, whichever
    // path wins — the promotion flavour is not part of the chip's identity.
    it.each([
      ["auto-continue first", [0, 1]],
      ["drain hint first", [1, 0]],
    ])(
      "stores one fallback when the text-bearing hint's GET overlaps the new-id GET (%s)",
      async (_label, order) => {
        const gets = deferPendingGets();
        const { view, getMessages, rerender } = setupHook([
          assistantMessage(0),
        ]);
        act(() => {
          view.result.current.queueMessage("follow up");
        });

        await act(async () => {
          rerender([assistantMessage(0), continuationMessage()]);
        });
        expect(gets.count()).toBe(1);

        await act(async () => {
          rerender([
            assistantMessage(0),
            continuationMessage([{ text: "follow up" }]),
          ]);
        });
        expect(gets.count()).toBe(2);

        for (const index of order) await gets.resolveDrained(index);

        expect(view.result.current.queuedMessages).toEqual([]);
        expect(storedFallbacks(getMessages())).toHaveLength(1);
        expect(renderedUserRows(getMessages()).map((row) => row.text)).toEqual([
          "follow up",
        ]);
        const messages = getMessages();
        expect(messages[messages.length - 1].id).toBe("assistant-continuation");
      },
    );

    it.each([
      ["auto-continue first", [0, 1]],
      ["count-only hint first", [1, 0]],
    ])(
      "renders one follow-up when a count-only hint's GET overlaps the new-id GET (%s)",
      async (_label, order) => {
        const gets = deferPendingGets();
        const { view, getMessages, rerender } = setupHook([
          assistantMessage(0),
        ]);
        act(() => {
          view.result.current.queueMessage("follow up");
        });

        await act(async () => {
          rerender([assistantMessage(0), continuationMessage()]);
        });
        await act(async () => {
          rerender([assistantMessage(0), continuationMessage(["count-only"])]);
        });
        expect(gets.count()).toBe(2);

        for (const index of order) await gets.resolveDrained(index);

        expect(view.result.current.queuedMessages).toEqual([]);
        expect(storedFallbacks(getMessages())).toHaveLength(1);
        expect(renderedUserRows(getMessages()).map((row) => row.text)).toEqual([
          "follow up",
        ]);
      },
    );

    it("renders one follow-up when the backstop poll overlaps the new-id GET", async () => {
      vi.useFakeTimers();
      try {
        const gets = deferPendingGets();
        const { view, getMessages, rerender } = setupHook([
          assistantMessage(0),
        ]);
        act(() => {
          view.result.current.queueMessage("follow up");
        });

        await act(async () => {
          rerender([assistantMessage(0), continuationMessage()]);
        });
        expect(gets.count()).toBe(1);

        // No hint ever arrives; the backstop fires while the first GET is
        // still open.
        await act(async () => {
          await vi.advanceTimersByTimeAsync(10_000);
        });
        expect(gets.count()).toBe(2);

        await gets.resolveDrained(1);
        await gets.resolveDrained(0);

        expect(view.result.current.queuedMessages).toEqual([]);
        expect(storedFallbacks(getMessages())).toHaveLength(1);
        expect(renderedUserRows(getMessages()).map((row) => row.text)).toEqual([
          "follow up",
        ]);
      } finally {
        vi.useRealTimers();
      }
    });

    it("keeps two bubbles for two genuinely repeated follow-ups drained together", async () => {
      const gets = deferPendingGets();
      const { view, getMessages, rerender } = setupHook([assistantMessage(0)]);
      act(() => {
        view.result.current.queueMessage("continue");
        view.result.current.queueMessage("continue");
      });

      await act(async () => {
        rerender([assistantMessage(0), continuationMessage()]);
      });
      await act(async () => {
        rerender([assistantMessage(0), continuationMessage(["count-only"])]);
      });
      expect(gets.count()).toBe(2);

      await gets.resolveDrained(0);
      await gets.resolveDrained(1);

      expect(view.result.current.queuedMessages).toEqual([]);
      expect(storedFallbacks(getMessages())).toHaveLength(2);
      expect(renderedUserRows(getMessages()).map((row) => row.text)).toEqual([
        "continue",
        "continue",
      ]);
    });
  });

  it("does not promote when the backend buffer count still covers the local chips", async () => {
    mockGetPending.mockResolvedValue({
      status: 200,
      data: { count: 1, messages: ["follow up"] },
      headers: new Headers(),
    } as Awaited<ReturnType<typeof getV2GetPendingMessages>>);

    const { view, getMessages } = setupHook([assistantMessage(0)]);
    act(() => {
      view.result.current.queueMessage("follow up");
    });

    await act(async () => {
      view.rerender({ messages: [assistantMessage(1)] });
    });

    await waitFor(() => expect(mockGetPending).toHaveBeenCalledWith("s1"));
    expect(
      getMessages().some((m) =>
        m.id.startsWith("promoted-midturn-pending-chip-"),
      ),
    ).toBe(false);
    expect(view.result.current.queuedMessages).toEqual(["follow up"]);
  });

  it("does not promote the previous session's chips after a session switch", async () => {
    let current: Messages = [assistantMessage(0)];
    const setMessages = vi.fn(
      (updater: Messages | ((prev: Messages) => Messages)) => {
        current = typeof updater === "function" ? updater(current) : updater;
      },
    );

    const view = renderHook(
      ({ messages, sessionId }) =>
        useCopilotPendingChips({
          sessionId,
          status: "streaming",
          messages,
          setMessages,
        }),
      {
        initialProps: {
          messages: [assistantMessage(0)],
          sessionId: "s1" as string,
        },
      },
    );

    act(() => {
      view.result.current.queueMessage("old session chip");
    });
    expect(view.result.current.queuedMessages).toEqual(["old session chip"]);

    // Switch to s2 with a HIGHER drain-hint count than s1's baseline while
    // the old chip is still queued for this commit.  The drain effect must
    // re-baseline on the session change and NOT promote the stale chip into
    // the new chat.
    await act(async () => {
      view.rerender({ messages: [assistantMessage(1)], sessionId: "s2" });
    });

    // The session-switch peek rebases the strip to the new (empty) session.
    await waitFor(() => {
      expect(view.result.current.queuedMessages).toEqual([]);
    });

    // No mid-turn promotion of the old chip leaked into the new session.
    expect(
      current.some((m) => m.id.startsWith("promoted-midturn-pending-chip-")),
    ).toBe(false);
  });

  it("backstop poll promotes a chip even without an SSE hint", async () => {
    vi.useFakeTimers();
    try {
      const { view, getMessages } = setupHook([assistantMessage(0)]);
      act(() => {
        view.result.current.queueMessage("follow up");
      });

      // No hint part is ever added; the slow backstop interval must still
      // reconcile against the drained buffer.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });

      expect(mockGetPending).toHaveBeenCalledWith("s1");
      expect(
        getMessages().some((m) =>
          m.id.startsWith("promoted-midturn-pending-chip-"),
        ),
      ).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  describe("restoring the buffer on session load", () => {
    /** Hold every peek GET open. `resolveAll` answers each with the same
     *  one-message buffer, the way the backend does while a follow-up is
     *  still queued; `resolveWith` answers one peek with a given buffer. */
    function deferBufferPeeks(text: string) {
      const resolvers: Array<(messages: string[]) => void> = [];
      mockGetPending.mockImplementation(
        () =>
          new Promise((resolve) => {
            resolvers.push((messages) =>
              resolve({
                status: 200,
                data: { count: messages.length, messages },
                headers: new Headers(),
              } as Awaited<ReturnType<typeof getV2GetPendingMessages>>),
            );
          }),
      );
      return {
        count: () => resolvers.length,
        resolveAll: () =>
          act(async () => {
            resolvers.splice(0).forEach((resolve) => resolve([text]));
          }),
        resolveWith: (index: number, messages: string[]) =>
          act(async () => {
            resolvers[index](messages);
          }),
      };
    }

    it("shows the queued follow-up once when two peeks overlap on load", async () => {
      const peeks = deferBufferPeeks("follow up");
      const setMessages = vi.fn();
      const view = renderHook(
        ({ status }) =>
          useCopilotPendingChips({
            sessionId: "s1",
            status,
            messages: [],
            setMessages,
          }),
        { initialProps: { status: "ready" as "ready" | "error" } },
      );
      // A second idle edge lands before the first peek has answered.
      view.rerender({ status: "error" });
      expect(peeks.count()).toBe(2);

      await peeks.resolveAll();

      // Both peeks report the same buffer: the strip must show that one
      // message, not one copy per peek. A doubled strip is what made the
      // next new-assistant reconciliation promote the extra copy above the
      // running tool chain while the backend still held the message.
      await waitFor(() =>
        expect(view.result.current.queuedMessages).toEqual(["follow up"]),
      );
      expect(setMessages).not.toHaveBeenCalled();
    });

    it("shows the queued follow-up once under a Strict Mode mount", async () => {
      const peeks = deferBufferPeeks("follow up");
      const setMessages = vi.fn();
      // The dev server mounts every effect twice, so the load peek fires
      // twice with the same empty snapshot.
      const view = renderHook(
        () =>
          useCopilotPendingChips({
            sessionId: "s1",
            status: "ready",
            messages: [],
            setMessages,
          }),
        { wrapper: StrictMode },
      );
      expect(peeks.count()).toBe(2);

      await peeks.resolveAll();

      await waitFor(() =>
        expect(view.result.current.queuedMessages).toEqual(["follow up"]),
      );
    });

    it("keeps a message typed between two overlapping peeks", async () => {
      const peeks = deferBufferPeeks("typed meanwhile");
      const setMessages = vi.fn();
      const view = renderHook(
        ({ status }) =>
          useCopilotPendingChips({
            sessionId: "s1",
            status,
            messages: [],
            setMessages,
          }),
        { initialProps: { status: "ready" as "ready" | "error" } },
      );
      act(() => {
        view.result.current.queueMessage("typed meanwhile");
      });
      view.rerender({ status: "error" });
      expect(peeks.count()).toBe(2);

      // The newer peek already sees the message on the server; the older
      // one predates it and answers last. Its stale, empty buffer must not
      // wipe the message the newer peek restored — and the snapshot it
      // filters with must be its own, not the newer peek's, or it would
      // treat the typed entry as already on the server and drop it.
      await peeks.resolveWith(1, ["typed meanwhile"]);
      await peeks.resolveWith(0, []);

      await waitFor(() =>
        expect(view.result.current.queuedMessages).toEqual(["typed meanwhile"]),
      );
    });

    it("restores a buffer the turn-start peek finds when it superseded the load peek", async () => {
      const peeks = deferBufferPeeks("from before");
      const setMessages = vi.fn();
      const view = renderHook(
        ({ status }) =>
          useCopilotPendingChips({
            sessionId: "s1",
            status,
            messages: [],
            setMessages,
          }),
        { initialProps: { status: "ready" as ChatStatus } },
      );
      // The user sends a prompt before the load peek has answered.
      view.rerender({ status: "submitted" });
      view.rerender({ status: "streaming" });
      expect(peeks.count()).toBe(2);

      // The turn-start peek answers first and wins; the load peek's answer
      // is stale and dropped. The buffered message must still reach the
      // strip, or nothing would poll for it during the whole turn.
      await peeks.resolveWith(1, ["from before"]);
      await peeks.resolveWith(0, ["from before"]);

      await waitFor(() =>
        expect(view.result.current.queuedMessages).toEqual(["from before"]),
      );
      expect(setMessages).not.toHaveBeenCalled();
    });

    it("restores only the buffered messages the strip does not already hold", async () => {
      const peeks = deferBufferPeeks("from before");
      const setMessages = vi.fn();
      const view = renderHook(
        ({ status }) =>
          useCopilotPendingChips({
            sessionId: "s1",
            status,
            messages: [],
            setMessages,
          }),
        { initialProps: { status: "ready" as ChatStatus } },
      );
      view.rerender({ status: "submitted" });
      act(() => {
        view.result.current.queueMessage("typed now");
      });
      view.rerender({ status: "streaming" });
      expect(peeks.count()).toBe(2);

      await peeks.resolveWith(1, ["from before", "typed now"]);
      await peeks.resolveWith(0, ["from before"]);

      await waitFor(() =>
        expect(view.result.current.queuedMessages).toEqual([
          "from before",
          "typed now",
        ]),
      );
      expect(setMessages).not.toHaveBeenCalled();
    });

    it("keeps a message typed during the peek window", async () => {
      const peeks = deferBufferPeeks("from server");
      const setMessages = vi.fn();
      const view = renderHook(() =>
        useCopilotPendingChips({
          sessionId: "s1",
          status: "ready",
          messages: [],
          setMessages,
        }),
      );
      expect(peeks.count()).toBe(1);
      act(() => {
        view.result.current.queueMessage("typed meanwhile");
      });

      await peeks.resolveAll();

      await waitFor(() =>
        expect(view.result.current.queuedMessages).toEqual([
          "from server",
          "typed meanwhile",
        ]),
      );
    });
  });
});
