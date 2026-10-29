import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";

import { dropQueuedSendBubbles } from "../dropQueuedSendBubbles";

function row(id: string, role: UIMessage["role"], text: string): UIMessage {
  return { id, role, parts: [{ type: "text", text }] };
}

const STORED = [
  row("sess-seq-0", "user", "Plan my week"),
  row("sess-seq-1", "assistant", "Here is your week."),
];

describe("dropQueuedSendBubbles", () => {
  it("drops the bubble of a send the queue holds", () => {
    const messages = [...STORED, row("local-1", "user", "Status?")];

    expect(dropQueuedSendBubbles(messages, ["Status?"])).toEqual(STORED);
  });

  it("keeps an earlier answered bubble with the same words and drops the newest", () => {
    const answered = row("local-1", "user", "ok");
    const answer = row("local-2", "assistant", "Done.");
    const queued = row("local-3", "user", "ok");

    expect(
      dropQueuedSendBubbles([...STORED, answered, answer, queued], ["ok"]),
    ).toEqual([...STORED, answered, answer]);
  });

  it("leaves stored rows and promoted chips alone", () => {
    const promoted = row("promoted-midturn-pending-chip-1", "user", "ok");
    const messages = [...STORED, promoted];

    expect(dropQueuedSendBubbles(messages, ["Plan my week", "ok"])).toBe(
      messages,
    );
  });
});
