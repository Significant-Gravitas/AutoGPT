import { describe, expect, it } from "vitest";
import { isTeamEmail } from "../team-email-policy";

describe("isTeamEmail", () => {
  it.each([
    ["someone@agpt.co", true],
    ["Someone@AGPT.CO", true],
    ["someone@previews.agpt.co", false],
    ["someone@agpt.com", false],
    ["agpt.co@evil.com", false],
    ["someone@agpt.co.evil.com", false],
    ["no-at-sign", false],
  ])("%s → %s", (email, expected) => {
    expect(isTeamEmail(email)).toBe(expected);
  });
});
