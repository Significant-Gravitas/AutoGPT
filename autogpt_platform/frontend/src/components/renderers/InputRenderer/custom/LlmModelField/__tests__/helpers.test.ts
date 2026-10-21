import { describe, expect, test } from "vitest";
import { compareModelEntries } from "../helpers";

function entry(title: string, provider: string, name: string) {
  return { title, entries: [{ provider, name }] };
}

describe("model menu ordering", () => {
  test("places MiniMax Flash before M3 without changing other providers", () => {
    const m3 = entry("MiniMax M3", "minimax", "MiniMax-M3");
    const flash = entry("MiniMax M3.1 Flash", "minimax", "MiniMax-M3.1-Flash");
    expect([m3, flash].sort(compareModelEntries)).toEqual([flash, m3]);
    const alpha = entry("Alpha", "other", "alpha");
    const beta = entry("Beta", "other", "beta");
    expect([beta, alpha].sort(compareModelEntries)).toEqual([alpha, beta]);
    expect(compareModelEntries(m3, m3)).toBe(0);
  });
});
