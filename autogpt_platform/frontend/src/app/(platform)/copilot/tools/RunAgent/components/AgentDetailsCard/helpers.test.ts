import type { RJSFSchema } from "@rjsf/utils";
import { describe, expect, it } from "vitest";
import { buildInputSchema, extractDefaults, isFormValid } from "./helpers";

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

  it("returns true for empty schema with empty data", () => {
    const schema: RJSFSchema = {
      type: "object",
      properties: {},
    };
    expect(isFormValid(schema, {})).toBe(true);
  });
});
