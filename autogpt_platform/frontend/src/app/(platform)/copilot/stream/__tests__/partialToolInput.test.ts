import { describe, expect, it } from "vitest";

import { readPartialToolInput } from "../partialToolInput";

describe("readPartialToolInput", () => {
  it.each([
    ["", undefined],
    ["[1, 2", undefined],
    ['"just a string', undefined],
    ["{", {}],
    ['{"ur', {}],
    ['{"url"', {}],
    ['{"url": ', { url: null }],
    ['{"url": "a.c', { url: "a.c" }],
    ['{"url": "a\\', { url: "a" }],
    ['{"url": "a.com", ', { url: "a.com" }],
    ['{"url": "a.com", "n": 4', { url: "a.com", n: 4 }],
    ['{"url": "a, b", "opts": {"x": [1, ', { url: "a, b", opts: null }],
    ['{"q": "say \\"hi\\", ok", "k": "v"}', { q: 'say "hi", ok', k: "v" }],
    ['  {"a": {"b": 1}, "c": tr', { a: { b: 1 }, c: null }],
  ])("reads %j as %j", (text, expected) => {
    expect(readPartialToolInput(text)).toEqual(expected);
  });
});
