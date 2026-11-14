import type { UIMessage } from "ai";

import { groupTurns } from "../turns";

function message(id: string, role: UIMessage["role"], text: string): UIMessage {
  return { id, role, parts: [{ type: "text", text }] };
}

describe("groupTurns", () => {
  it("starts a turn at each user message and keeps the replies with it", () => {
    const turns = groupTurns([
      message("a", "assistant", "Hi"),
      message("b", "user", "Question"),
      message("c", "assistant", "Answer"),
      message("d", "assistant", "More"),
      message("e", "user", "Next"),
    ]);
    expect(turns.map((turn) => turn.items.map((item) => item.message.id))).toEqual([
      ["a"],
      ["b", "c", "d"],
      ["e"],
    ]);
  });

  it("gives each message the text of the one after it", () => {
    const turns = groupTurns([
      message("a", "assistant", "Pick one"),
      message("b", "user", "This one"),
    ]);
    expect(turns[0].items[0].replyText).toBe("This one");
    expect(turns[1].items[0].replyText).toBeNull();
  });
});
