import { describe, expect, test } from "vitest";
import { expertInitials } from "./helpers";

describe("expertInitials", () => {
  test.each([
    ["", "?"],
    ["Otto", "O"],
    ["Nova Ray", "NR"],
    ["  juno   star  extra ", "JS"],
  ])("converts %j to %s", (name, expected) => {
    expect(expertInitials(name)).toBe(expected);
  });
});
