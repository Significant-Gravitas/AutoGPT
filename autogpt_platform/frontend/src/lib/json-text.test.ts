import { describe, expect, it } from "vitest";
import {
  isJsonTextFor,
  parseJsonExact,
  parseJsonExactWithIntegers,
} from "./json-text";

describe("isJsonTextFor", () => {
  it("matches the same value however it is formatted", () => {
    expect(isJsonTextFor('{"a":1}', { a: 1 })).toBe(true);
    expect(isJsonTextFor('{\n  "a": 1\n}', { a: 1 })).toBe(true);
    expect(isJsonTextFor("[1, 2]", [1, 2])).toBe(true);
  });

  it("does not match a different value", () => {
    expect(isJsonTextFor('{"a":1}', { a: 2 })).toBe(false);
    expect(isJsonTextFor("[1, 2]", [2, 1])).toBe(false);
  });

  it("does not match text that is not JSON yet", () => {
    expect(isJsonTextFor('{"a":', { a: 1 })).toBe(false);
  });

  it("matches empty text only to no value", () => {
    expect(isJsonTextFor("  ", undefined)).toBe(true);
    expect(isJsonTextFor("", null)).toBe(true);
    expect(isJsonTextFor("", {})).toBe(false);
  });

  // The stored value would be a rounded copy, so the field must show it
  // rather than keep digits that will not be saved.
  it("does not match numbers JSON parsing can't hold exactly", () => {
    const id = "1234567890123456789";
    expect(
      isJsonTextFor(
        `{"channel_id": ${id}}`,
        JSON.parse(`{"channel_id": ${id}}`),
      ),
    ).toBe(false);
    expect(isJsonTextFor("[1e400]", [Infinity])).toBe(false);
  });

  it("still matches safe numbers and digits inside strings", () => {
    expect(
      isJsonTextFor("[9007199254740991, -2.5e3]", [9007199254740991, -2500]),
    ).toBe(true);
    expect(
      isJsonTextFor('{"id": "1234567890123456789"}', {
        id: "1234567890123456789",
      }),
    ).toBe(true);
  });

  // A big integer is kept as its digits, so the text that typed it matches.
  it("matches a big integer kept as its digit string", () => {
    expect(
      isJsonTextFor('{"id": 9007199254740993}', { id: "9007199254740993" }),
    ).toBe(true);
    expect(isJsonTextFor("[9007199254740992]", ["9007199254740992"])).toBe(
      true,
    );
  });

  // "-0" is on the way to "-0.5" and is stored exactly; it reads back as "0".
  it("matches negative zero", () => {
    expect(isJsonTextFor("[-0]", [-0])).toBe(true);
  });
});

describe("parseJsonExact", () => {
  it("keeps integers beyond 2^53 as their digit string", () => {
    expect(
      parseJsonExact(
        '{"a": 9007199254740993, "b": [-9223372036854775807], "c": 42}',
      ),
    ).toEqual({ a: "9007199254740993", b: ["-9223372036854775807"], c: 42 });
  });

  it("parses everything else like JSON.parse", () => {
    const text =
      '{"s": "id 9007199254740993 \\" 12345678901234567890", "f": 9007199254740993.5, "e": 1e20, "n": null, "t": true}';
    expect(parseJsonExact(text)).toEqual(JSON.parse(text));
  });

  it("throws on invalid JSON like JSON.parse", () => {
    expect(() => parseJsonExact('{"a": 9007199254740993')).toThrow();
  });

  it("rejects leading-zero numbers like JSON.parse", () => {
    const text = "[09007199254740993]";
    expect(() => JSON.parse(text)).toThrow();
    expect(() => parseJsonExact(text)).toThrow();
    expect(parseJsonExact("[0, -0, 0.5, 0e20]")).toEqual([0, -0, 0.5, 0]);
  });
});

describe("parseJsonExactWithIntegers", () => {
  const big = "9007199254740995";

  it("gives the path of each big integer that was a JSON number", () => {
    const { value, integers } = parseJsonExactWithIntegers(
      `{"a/b": {"list": [1, ${big}]}, "label": "${big}", "n": -${big}}`,
    );
    expect(value).toEqual({
      "a/b": { list: [1, big] },
      label: big,
      n: `-${big}`,
    });
    expect(integers).toEqual([
      { path: ["a/b", "list", "1"], digits: big },
      { path: ["n"], digits: `-${big}` },
    ]);
  });

  it("gives a bare top-level integer the empty path", () => {
    expect(parseJsonExactWithIntegers(big)).toEqual({
      value: big,
      integers: [{ path: [], digits: big }],
    });
  });

  it("keeps a string of a control character and digits a string", () => {
    const text = `{"s": "\\u0000${big}", "t": "\\u0001${big}", "n": ${big}}`;
    const { value, integers } = parseJsonExactWithIntegers(text);
    expect(value).toEqual({ s: `\u0000${big}`, t: `\u0001${big}`, n: big });
    expect(integers).toEqual([{ path: ["n"], digits: big }]);
  });

  it("reads floats and short integers without the exact pass", () => {
    for (const text of [
      '{"f": 0.30000000000000004, "g": 1.2345678901234567e30}',
      '{"n": 123456789012345, "f": 12345678901234567.5}',
    ]) {
      expect(parseJsonExactWithIntegers(text)).toEqual({
        value: JSON.parse(text),
        integers: [],
      });
    }
  });
});
