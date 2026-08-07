import { describe, expect, it } from "vitest";
import { buildSampleEvents } from "../sampleScript";

describe("buildSampleEvents", () => {
  it("builds a complete, timed conversation", () => {
    const events = buildSampleEvents();
    const kinds = new Set(events.map((event) => event.kind));

    expect(events[0]).toMatchObject({ kind: "user", delay: 0 });
    expect(kinds).toEqual(
      new Set([
        "user",
        "await-user",
        "status",
        "assistant-start",
        "text-start",
        "text-delta",
        "reasoning-start",
        "reasoning-delta",
        "reasoning-done",
        "tool-start",
        "tool-output",
        "tool-error",
      ]),
    );
    expect(
      events.find(
        (event) =>
          event.kind === "tool-start" && event.toolCallId === "t-todo-1",
      ),
    ).toMatchObject({ delay: 875 });
    expect(events.at(-1)).toMatchObject({
      kind: "text-delta",
      messageId: "sample-assistant-2",
    });
  });
});
