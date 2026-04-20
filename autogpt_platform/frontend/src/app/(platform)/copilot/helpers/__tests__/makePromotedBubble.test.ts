import { describe, expect, it } from "vitest";

import { makePromotedUserBubble } from "../makePromotedBubble";

describe("makePromotedUserBubble", () => {
  it("joins texts with double newlines", () => {
    const b = makePromotedUserBubble(["first", "second"], "midturn", "abc");
    expect(b.role).toBe("user");
    expect(b.parts).toHaveLength(1);
    expect(b.parts[0]).toMatchObject({
      type: "text",
      text: "first\n\nsecond",
      state: "done",
    });
  });

  it("encodes the prefix + suffix in the id for dedup", () => {
    const a = makePromotedUserBubble(["x"], "auto-continue", "assistant-123");
    const b = makePromotedUserBubble(["x"], "midturn", "uuid-456");
    expect(a.id).toBe("promoted-auto-continue-assistant-123");
    expect(b.id).toBe("promoted-midturn-uuid-456");
  });

  it("handles a single text without joining", () => {
    const b = makePromotedUserBubble(["only"], "midturn", "s");
    expect(b.parts[0]).toMatchObject({ text: "only" });
  });

  it("yields empty-text bubble if given an empty list", () => {
    const b = makePromotedUserBubble([], "midturn", "s");
    expect(b.parts[0]).toMatchObject({ text: "" });
  });
});
