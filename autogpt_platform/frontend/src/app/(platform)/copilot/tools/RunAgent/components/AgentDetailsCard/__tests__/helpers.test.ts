import type { RJSFSchema } from "@rjsf/utils";
import { describe, expect, it } from "vitest";
import { buildInputSchema, extractDefaults, isFormValid } from "../helpers";

describe("buildInputSchema", () => {
  it("returns null for falsy input", () => {
    expect(buildInputSchema(null)).toBeNull();
    expect(buildInputSchema(undefined)).toBeNull();
    expect(buildInputSchema("")).toBeNull();
  });

  it("returns null for empty properties object", () => {
    expect(buildInputSchema({})).toBeNull();
  });

  it("returns the schema when properties exist", () => {
    const schema = { name: { type: "string" as const } };
    expect(buildInputSchema(schema)).toBe(schema);
  });
});

describe("extractDefaults", () => {
  it("returns an empty object when no properties exist", () => {
    expect(extractDefaults({})).toEqual({});
    expect(extractDefaults({ properties: null as never })).toEqual({});
  });

  it("extracts default values from property definitions", () => {
    const schema: RJSFSchema = {
      properties: {
        name: { type: "string", default: "Alice" },
        age: { type: "number", default: 30 },
      },
    };
    expect(extractDefaults(schema)).toEqual({ name: "Alice", age: 30 });
  });

  it("falls back to the first example when no default is defined", () => {
    const schema: RJSFSchema = {
      properties: {
        query: { type: "string", examples: ["hello", "world"] },
      },
    };
    expect(extractDefaults(schema)).toEqual({ query: "hello" });
  });

  it("prefers default over examples", () => {
    const schema: RJSFSchema = {
      properties: {
        value: { type: "string", default: "def", examples: ["ex"] },
      },
    };
    expect(extractDefaults(schema)).toEqual({ value: "def" });
  });

  it("skips properties without default or examples", () => {
    const schema: RJSFSchema = {
      properties: {
        name: { type: "string" },
        title: { type: "string", default: "Mr." },
      },
    };
    expect(extractDefaults(schema)).toEqual({ title: "Mr." });
  });

  it("skips properties that are not objects", () => {
    const schema: RJSFSchema = {
      properties: {
        bad: true,
        alsobad: false,
      },
    };
    expect(extractDefaults(schema)).toEqual({});
  });
});

describe("isFormValid", () => {
  it("returns true for a valid form", () => {
    const schema: RJSFSchema = {
      type: "object",
      properties: {
        name: { type: "string" },
      },
    };
    expect(isFormValid(schema, { name: "Alice" })).toBe(true);
  });

  it("returns false when required fields are missing", () => {
    const schema: RJSFSchema = {
      type: "object",
      required: ["name"],
      properties: {
        name: { type: "string" },
      },
    };
    expect(isFormValid(schema, {})).toBe(false);
  });

  it("accepts an integer beyond 2^53 held as a digit string, unchanged", () => {
    const schema: RJSFSchema = {
      type: "object",
      required: ["n"],
      properties: {
        n: { type: "integer" },
        maybe: { anyOf: [{ type: "integer" }, { type: "null" }] },
      },
    };
    const formData = { n: "9007199254740993", maybe: "-9223372036854775807" };

    expect(isFormValid(schema, formData)).toBe(true);
    expect(formData).toEqual({
      n: "9007199254740993",
      maybe: "-9223372036854775807",
    });
  });

  it("still rejects non-digit strings for integer fields", () => {
    const schema: RJSFSchema = {
      type: "object",
      properties: { n: { type: "integer" } },
    };
    expect(isFormValid(schema, { n: "12abc" })).toBe(false);
    expect(isFormValid(schema, { n: "1.5" })).toBe(false);
  });

  it.each(["default", "const", "enum", "examples"])(
    "accepts a big integer in an input named %j",
    (name) => {
      const schema: RJSFSchema = {
        type: "object",
        properties: { [name]: { type: "integer" } },
      };
      expect(isFormValid(schema, { [name]: "9007199254740993" })).toBe(true);
    },
  );

  it("keeps a digit string valid when a oneOf's string branch accepts it", () => {
    const schema: RJSFSchema = {
      type: "object",
      properties: {
        id: { oneOf: [{ type: "integer" }, { type: "string" }] },
      },
    };
    expect(isFormValid(schema, { id: "9007199254740993" })).toBe(true);
    expect(isFormValid(schema, { id: "order-7" })).toBe(true);
    expect(isFormValid(schema, { id: 7 })).toBe(true);
  });

  it("lets a oneOf's integer branch take a big integer its string branch refuses", () => {
    const schema: RJSFSchema = {
      type: "object",
      properties: {
        id: { oneOf: [{ type: "integer" }, { type: "string", maxLength: 5 }] },
      },
    };
    expect(isFormValid(schema, { id: "9007199254740993" })).toBe(true);
    expect(isFormValid(schema, { id: "order-7777" })).toBe(false);
  });

  it("still needs exactly one match in a oneOf of integers", () => {
    const schema: RJSFSchema = {
      type: "object",
      properties: {
        n: {
          oneOf: [
            { type: "integer", maximum: 10 },
            { type: "integer", minimum: 0 },
          ],
        },
      },
    };
    expect(isFormValid(schema, { n: "9007199254740993" })).toBe(true);
    expect(isFormValid(schema, { n: "-9007199254740993" })).toBe(true);
    expect(isFormValid(schema, { n: 5 })).toBe(false);
    expect(isFormValid(schema, { n: "5" })).toBe(false);
  });

  it.each([
    ["below the minimum", { minimum: 0 }, "-9007199254740993", false],
    [
      "inside the bounds",
      { minimum: 0, maximum: 2 ** 63 },
      "9223372036854775807",
      true,
    ],
    [
      "one above a maximum of 2^53",
      { maximum: 2 ** 53 },
      "9007199254740993",
      false,
    ],
    [
      "one above an exclusive minimum of 2^53",
      { exclusiveMinimum: 2 ** 53 },
      "9007199254740993",
      true,
    ],
    [
      "odd, for a multipleOf of 2",
      { multipleOf: 2 },
      "9007199254740993",
      false,
    ],
    [
      "even, for a multipleOf of 2",
      { multipleOf: 2 },
      "9007199254740994",
      true,
    ],
  ])("checks a big integer %s exactly", (_, bounds, value, valid) => {
    const schema: RJSFSchema = {
      type: "object",
      properties: { n: { type: "integer", ...bounds } },
    };
    expect(isFormValid(schema, { n: value })).toBe(valid);
  });

  it("leaves string and number fields unchanged", () => {
    const schema: RJSFSchema = {
      type: "object",
      properties: {
        name: { type: "string", pattern: "^[a-z]+$" },
        ratio: { type: "number" },
      },
    };
    expect(isFormValid(schema, { name: "abc", ratio: 0.5 })).toBe(true);
    expect(isFormValid(schema, { name: "ABC" })).toBe(false);
    expect(isFormValid(schema, { ratio: "9007199254740993" })).toBe(false);
  });

  it("returns true for empty schema with empty data", () => {
    const schema: RJSFSchema = {
      type: "object",
      properties: {},
    };
    expect(isFormValid(schema, {})).toBe(true);
  });
});
