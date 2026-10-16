import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  advanceReply,
  beginReplyTracking,
  SETTLE_GRACE_MS,
  type ReplyStep,
  type ReplyTimingStart,
  type ReplyView,
} from "../replyTiming";

function text(id: string, role: UIMessage["role"], content: string) {
  return { id, role, parts: [{ type: "text", text: content }] } as UIMessage;
}

function toolCall(id: string) {
  return {
    id,
    role: "assistant",
    parts: [
      { type: "step-start" },
      {
        type: "tool-web_search",
        toolCallId: "call-1",
        state: "input-available",
        input: {},
      },
    ],
  } as UIMessage;
}

const HISTORY = [
  text("u0", "user", "earlier question"),
  text("a0", "assistant", "An earlier answer."),
];
const SENT = text("u1", "user", "new question");
const WORKING = [...HISTORY, SENT, toolCall("a1")];
const START: ReplyTimingStart = {
  startedAt: 1000,
  timeOrigin: performance.timeOrigin,
  isNewChat: false,
  isExpert: false,
};

function view(overrides: Partial<ReplyView>): ReplyView {
  return {
    sessionId: "session-1",
    messages: HISTORY,
    status: "ready",
    isSettled: true,
    isUserStopping: false,
    ...overrides,
  };
}

function streaming(messages: UIMessage[]) {
  return view({ messages, status: "streaming", isSettled: false });
}

function tracked() {
  const tracking = beginReplyTracking(START, view({}), "ai-sdk");
  if (!tracking) throw new Error("the send was not tracked");
  return tracking;
}

function stillFollowed(step: ReplyStep) {
  if (!step.tracking) throw new Error("the reply ended early");
  return step.tracking;
}

/** A reply that showed a tool card at 1100 and whose turn ended at 1500. */
function endedWithoutText() {
  const shown = advanceReply(tracked(), streaming(WORKING), 1100);
  return advanceReply(stillFollowed(shown), view({ messages: WORKING }), 1500);
}

describe("reply timing", () => {
  it("drops a send whose start was read on another page's clock", () => {
    const fromBeforeNavigation = {
      ...START,
      timeOrigin: performance.timeOrigin - 60_000,
    };

    expect(
      beginReplyTracking(fromBeforeNavigation, view({}), "ai-sdk"),
    ).toBeNull();
  });

  it("only counts what shows after the message just sent", () => {
    const step = advanceReply(tracked(), streaming([...HISTORY, SENT]), 1200);

    expect(step.record).toBeNull();
    expect(step.tracking).toMatchObject({ sentId: "u1", firstOutput: null });
  });

  it("gives the screen a moment to catch up with a turn that has ended", () => {
    const ended = endedWithoutText();
    expect(ended).toMatchObject({ record: null, recheckInMs: SETTLE_GRACE_MS });

    // The stream runtime reveals the last of its text a tick after the turn.
    const caughtUp = advanceReply(
      stillFollowed(ended),
      view({ messages: [...WORKING, text("a2", "assistant", "Done.")] }),
      1530,
    );
    expect(caughtUp.record).toMatchObject({
      outcome: "answered",
      firstOutput: { ms: 100, kind: "tool" },
      firstTextMs: 530,
    });
  });

  it("ends a turn without text where it ended once the grace has passed", () => {
    const step = advanceReply(
      stillFollowed(endedWithoutText()),
      view({ messages: WORKING }),
      1500 + SETTLE_GRACE_MS,
    );

    expect(step.record).toMatchObject({
      outcome: "no_text",
      endedAt: 1500,
      firstTextMs: null,
    });
  });

  it("keeps the turn's ending when the list is rebuilt under new ids", () => {
    const rebuilt = [
      ...HISTORY,
      text("seq-3", "user", "new question"),
      toolCall("seq-4"),
    ];
    const step = advanceReply(
      stillFollowed(endedWithoutText()),
      view({ messages: rebuilt }),
      1550,
    );

    expect(step.record).toMatchObject({ outcome: "no_text", endedAt: 1500 });
  });

  it("drops a send taken back before its turn began", () => {
    const sent = advanceReply(tracked(), streaming([...HISTORY, SENT]), 1100);
    const refused = advanceReply(
      stillFollowed(sent),
      view({ messages: HISTORY, status: "error" }),
      1200,
    );

    expect(refused).toEqual({ tracking: null, record: null });
  });
});
