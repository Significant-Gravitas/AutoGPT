import { describe, expect, it } from "vitest";
import { isTeamEmail } from "../team-email";

describe("isTeamEmail (client-safe)", () => {
  it.each([
    ["a@agpt.co", true],
    ["A@AGPT.CO", true],
    ["a@agpt.com", false],
    ["a@agpt.co.uk", false],
    ["a@agpt.company", false],
    ["a@previews.agpt.co", false],
    ["no-at-sign", false],
  ])("%s → %s", (email, expected) => {
    expect(isTeamEmail(email)).toBe(expected);
  });
});
