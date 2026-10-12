import type { MouseEvent } from "react";
import { describe, expect, it } from "vitest";
import { isPlainLeftClick } from "../helpers";

function click(overrides: Partial<MouseEvent> = {}) {
  return {
    button: 0,
    metaKey: false,
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    ...overrides,
  } as MouseEvent;
}

describe("isPlainLeftClick", () => {
  it("accepts an unmodified primary click", () => {
    expect(isPlainLeftClick(click())).toBe(true);
  });

  it.each([
    ["middle button", { button: 1 }],
    ["meta", { metaKey: true }],
    ["ctrl", { ctrlKey: true }],
    ["shift", { shiftKey: true }],
    ["alt", { altKey: true }],
  ])("rejects a %s click", (_, overrides) => {
    expect(isPlainLeftClick(click(overrides))).toBe(false);
  });
});
