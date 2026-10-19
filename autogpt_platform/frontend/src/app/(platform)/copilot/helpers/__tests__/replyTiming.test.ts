import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  abandonReply,
  advanceReply,
  beginReplyTracking,
  QUIET_TURN_GRACE_MS,
  SETTLE_GRACE_MS,
  type ReplyStep,
  type ReplyTimingStart,
  type ReplyView,
} from "../replyTiming";

function text(id: string, role: UIMessage["role"], content: string) {
  return { id, role, parts: [{ type: "text", text: content }] } as UIMessage;
}

function toolPart(callId: string) {
  return {
    type: "tool-web_search",
    toolCallId: callId,
    state: "output-available",
    input: {},
    output: {},
  };
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

function assistant(id: string, parts: unknown[]) {
  return { id, role: "assistant", parts } as UIMessage;
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
    isQueued: false,
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
    const followed = advanceReply(
      stillFollowed(endedWithoutText()),
      view({ messages: rebuilt }),
      1550,
    );
    const step = advanceReply(
      stillFollowed(followed),
      view({ messages: rebuilt }),
      1500 + SETTLE_GRACE_MS,
    );

    expect(step.record).toMatchObject({ outcome: "no_text", endedAt: 1500 });
  });

  it("times an answer that only shows once the list is rebuilt", () => {
    const rebuilt = [
      ...HISTORY,
      text("seq-3", "user", "new question"),
      text("seq-4", "assistant", "Done."),
    ];
    const step = advanceReply(
      stillFollowed(endedWithoutText()),
      view({ messages: rebuilt }),
      1550,
    );

    expect(step.record).toMatchObject({
      outcome: "answered",
      firstTextMs: 550,
    });
  });

  it("keeps a turn's ending when its reply is given up in the grace", () => {
    const step = abandonReply(stillFollowed(endedWithoutText()), 1600);

    expect(step.record).toMatchObject({ outcome: "no_text", endedAt: 1500 });
  });

  it("does not take narration folded between tool calls for the answer", () => {
    const narrated = assistant("a1", [
      { type: "step-start" },
      toolPart("call-1"),
      { type: "text", text: "Checking the docs next." },
      toolPart("call-2"),
      toolPart("call-3"),
    ]);
    const working = advanceReply(
      tracked(),
      streaming([...HISTORY, SENT, narrated]),
      1100,
    );
    expect(working.record).toBeNull();

    const answered = assistant("a1", [
      ...narrated.parts,
      { type: "text", text: "Here is what I found." },
    ]);
    const step = advanceReply(
      stillFollowed(working),
      streaming([...HISTORY, SENT, answered]),
      1400,
    );
    expect(step.record).toMatchObject({
      outcome: "answered",
      firstOutput: { ms: 100, kind: "tool" },
      firstTextMs: 400,
    });
  });

  it("counts narration before the first tool call as the answer's start", () => {
    const narrated = assistant("a1", [
      { type: "step-start" },
      { type: "text", text: "Let me look that up." },
      toolPart("call-1"),
    ]);
    const step = advanceReply(
      tracked(),
      streaming([...HISTORY, SENT, narrated]),
      1100,
    );

    expect(step.record).toMatchObject({
      outcome: "answered",
      firstTextMs: 100,
    });
  });

  it("keeps following a send queued behind the user's other running turns", () => {
    const queued = view({ messages: [...HISTORY, SENT], isQueued: true });
    const sent = advanceReply(
      { ...tracked(), hasReturned: true },
      queued,
      1100,
    );
    expect(stillFollowed(sent).ending).toBeNull();

    const step = advanceReply(
      stillFollowed(sent),
      streaming([...HISTORY, SENT, text("a1", "assistant", "Done.")]),
      9000,
    );
    expect(step.record).toMatchObject({
      outcome: "answered",
      wasQueued: true,
      firstTextMs: 8000,
    });
  });

  it("waits longer on a turn that showed nothing before it ends it", () => {
    const shownNothing = advanceReply(
      { ...tracked(), hasReturned: true },
      view({ messages: [...HISTORY, SENT] }),
      1100,
    );
    expect(shownNothing).toMatchObject({
      record: null,
      recheckInMs: QUIET_TURN_GRACE_MS,
    });

    // The chat reads as queued once its status has been fetched again.
    const queued = advanceReply(
      stillFollowed(shownNothing),
      view({ messages: [...HISTORY, SENT], isQueued: true }),
      1100 + SETTLE_GRACE_MS * 2,
    );
    expect(stillFollowed(queued).ending).toBeNull();
  });

  it("gives up a reply when the user's next message shows before its answer", () => {
    const followedUp = [
      ...HISTORY,
      SENT,
      toolCall("a1"),
      text("u2", "user", "and also this"),
      text("a2", "assistant", "Both done."),
    ];
    const step = advanceReply(tracked(), streaming(followedUp), 1300);

    expect(step.record).toMatchObject({
      outcome: "abandoned",
      endedAt: 1300,
      firstTextMs: null,
    });
  });

  it("reads past a promoted follow-up above the reply it was drained into", () => {
    const promoted = [
      ...HISTORY,
      SENT,
      text("promoted-midturn-u2", "user", "and also this"),
      text("a1", "assistant", "Both done."),
    ];
    const step = advanceReply(tracked(), streaming(promoted), 1300);

    expect(step.record).toMatchObject({
      outcome: "answered",
      firstTextMs: 300,
    });
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
