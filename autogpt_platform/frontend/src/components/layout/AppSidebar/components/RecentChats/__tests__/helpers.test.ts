import { describe, expect, it } from "vitest";

import { getNewChatHref } from "../helpers";

describe("getNewChatHref", () => {
  const active = new Set(["expert-maria", "expert/slash"]);

  it("sends the Otto group to the plain home page", () => {
    expect(getNewChatHref(null, active)).toBe("/home");
  });

  it("addresses an active expert and asks for a fresh thread", () => {
    expect(getNewChatHref("expert-maria", active)).toBe(
      "/home?expertId=expert-maria&new=1",
    );
  });

  it("encodes the expert id", () => {
    expect(getNewChatHref("expert/slash", active)).toBe(
      "/home?expertId=expert%2Fslash&new=1",
    );
  });

  it("gives fired experts no link at all", () => {
    expect(getNewChatHref("expert-max", active)).toBeNull();
  });
});
