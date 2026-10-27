import { describe, expect, it } from "vitest";
import { reuseUnchanged } from "./reuseUnchanged";

describe("reuseUnchanged", () => {
  it("returns the previous value when the next one is deep-equal", () => {
    const previous = {
      status: "COMPLETED",
      output: { result: [1, { nested: "a" }] },
      at: new Date("2026-01-05T09:00:00.000Z"),
    };
    const next = {
      status: "COMPLETED",
      output: { result: [1, { nested: "a" }] },
      at: new Date("2026-01-05T09:00:00.000Z"),
    };

    expect(reuseUnchanged(previous, next)).toBe(previous);
  });

  it("keeps equal subtrees and replaces only the changed ones", () => {
    const previous = {
      input: { prompt: "hello", options: { temperature: 0 } },
      output: { result: ["first"] },
    };
    const next = {
      input: { prompt: "hello", options: { temperature: 0 } },
      output: { result: ["first", "second"] },
    };

    const merged = reuseUnchanged(previous, next);

    expect(merged).not.toBe(previous);
    expect(merged).toEqual(next);
    expect(merged.input).toBe(previous.input);
    expect(merged.output).not.toBe(previous.output);
    expect(merged.output.result).toEqual(["first", "second"]);
  });

  it("keeps unchanged array items when an item changes", () => {
    const first = { id: "a", value: 1 };
    const previous = [first, { id: "b", value: 2 }];
    const next = [
      { id: "a", value: 1 },
      { id: "b", value: 3 },
    ];

    const merged = reuseUnchanged(previous, next);

    expect(merged).not.toBe(previous);
    expect(merged[0]).toBe(first);
    expect(merged[1]).not.toBe(previous[1]);
    expect(merged[1]).toEqual({ id: "b", value: 3 });
  });

  it("treats added, removed and undefined keys as changes", () => {
    const previous: Record<string, unknown> = { a: 1 };

    expect(reuseUnchanged(previous, { a: 1, b: 2 })).not.toBe(previous);
    const wider = { a: 1, b: 2 };
    expect(reuseUnchanged(wider, previous)).not.toBe(wider);
    expect(reuseUnchanged(wider, previous)).toEqual({ a: 1 });
    expect(reuseUnchanged(previous, { a: 1, b: undefined })).not.toBe(previous);
    expect(reuseUnchanged({ b: undefined }, { c: undefined })).toEqual({
      c: undefined,
    });
  });

  it("compares dates by time", () => {
    const previous = new Date("2026-01-05T09:00:00.000Z");

    expect(reuseUnchanged(previous, new Date(previous.getTime()))).toBe(
      previous,
    );
    const later = new Date("2026-01-05T09:00:01.000Z");
    expect(reuseUnchanged(previous, later)).toBe(later);
  });

  it("returns the next value when the types differ", () => {
    expect(reuseUnchanged({ a: 1 }, [1])).toEqual([1]);
    expect(reuseUnchanged([1], { 0: 1 })).toEqual({ 0: 1 });
    expect(reuseUnchanged("1", 1)).toBe(1);
    expect(reuseUnchanged(null, {})).toEqual({});
    expect(reuseUnchanged({}, null)).toBeNull();
  });

  it("only reuses other objects when they are the same reference", () => {
    const map = new Map([["a", 1]]);
    const sameContent = new Map([["a", 1]]);

    expect(reuseUnchanged(map, sameContent)).toBe(sameContent);
    const holder = { map };
    expect(reuseUnchanged(holder, { map })).toBe(holder);
  });

  it("treats NaN as equal to NaN", () => {
    const previous = { value: Number.NaN };
    expect(reuseUnchanged(previous, { value: Number.NaN })).toBe(previous);
  });
});
