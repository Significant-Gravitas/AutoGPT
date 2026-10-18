import { getFirstMatchingOption, StrictRJSFSchema } from "@rjsf/utils";
import { describe, expect, it } from "vitest";
import { customValidator } from "../../../utils/custom-validator";
import { getDefaultTypeIndex, stripEmptyProperties } from "../helpers";

describe("stripEmptyProperties", () => {
  it("removes an empty properties object without mutating the input", () => {
    const option: StrictRJSFSchema = {
      type: "object",
      title: "Object",
      additionalProperties: true,
      properties: {},
    };

    const [result] = stripEmptyProperties([option]);

    expect(result).not.toHaveProperty("properties");
    expect(result).toEqual({
      type: "object",
      title: "Object",
      additionalProperties: true,
    });
    expect(option.properties).toEqual({});
  });

  it("keeps options with non-empty or missing properties unchanged", () => {
    const withProps: StrictRJSFSchema = {
      type: "object",
      properties: { name: { type: "string" } },
    };
    const primitive: StrictRJSFSchema = { type: "string" };

    const result = stripEmptyProperties([withProps, primitive]);

    expect(result[0]).toBe(withProps);
    expect(result[1]).toBe(primitive);
  });

  it("preserves option order and length", () => {
    const options: StrictRJSFSchema[] = [
      { type: "string" },
      { type: "object", additionalProperties: true, properties: {} },
      { type: "null" },
    ];

    const result = stripEmptyProperties(options);

    expect(result).toHaveLength(3);
    expect(result.map((o) => o.type)).toEqual(["string", "object", "null"]);
  });

  it("lets getFirstMatchingOption match without building an empty anyOf", () => {
    const options: StrictRJSFSchema[] = [
      { type: "string" },
      { type: "object", additionalProperties: true, properties: {} },
    ];

    const index = getFirstMatchingOption(
      customValidator,
      {},
      stripEmptyProperties(options),
      {},
    );

    expect(index).toBe(1);
  });
});

describe("getDefaultTypeIndex", () => {
  it("prefers string over other types", () => {
    expect(
      getDefaultTypeIndex([
        { type: "null" },
        { type: "object" },
        { type: "string" },
      ]),
    ).toBe(2);
  });

  it("falls back to the first non-null option", () => {
    expect(getDefaultTypeIndex([{ type: "null" }, {}])).toBe(1);
  });
});
