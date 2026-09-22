import { describe, expect, it } from "vitest";

import { makePromotedUserBubble } from "../makePromotedBubble";

describe("makePromotedUserBubble", () => {
  it("renders one bubble per chip with the chip text verbatim", () => {
    const b = makePromotedUserBubble("first", "midturn", "abc");
    expect(b.role).toBe("user");
    expect(b.parts).toHaveLength(1);
    expect(b.parts[0]).toMatchObject({
      type: "text",
      text: "first",
      state: "done",
    });
  });

  it("encodes the prefix + suffix in the id for dedup", () => {
    const a = makePromotedUserBubble("x", "auto-continue", "assistant-123");
    const b = makePromotedUserBubble("x", "midturn", "uuid-456");
    expect(a.id).toBe("promoted-auto-continue-assistant-123");
    expect(b.id).toBe("promoted-midturn-uuid-456");
  });
});
