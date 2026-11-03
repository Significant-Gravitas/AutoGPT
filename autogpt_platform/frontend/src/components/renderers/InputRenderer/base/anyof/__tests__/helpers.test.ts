import { getFirstMatchingOption, StrictRJSFSchema } from "@rjsf/utils";
import { describe, expect, it, vi } from "vitest";
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

  it("preserves frozen schemas and their nested constraints", () => {
    const patternProperties = Object.freeze({
      "^name": Object.freeze({ type: "string" as const }),
    });
    const option: StrictRJSFSchema = Object.freeze({
      type: "object",
      properties: Object.freeze({}),
      additionalProperties: false,
      patternProperties,
    });
    const options = [option, { type: "null" } as StrictRJSFSchema];
    Object.freeze(options);

    const result = stripEmptyProperties(options);

    expect(result).not.toBe(options);
    expect(result[0]).not.toBe(option);
    expect(result[0].patternProperties).toBe(patternProperties);
    expect(result[0].additionalProperties).toBe(false);
    expect(option.properties).toEqual({});
    expect(result[1]).toBe(options[1]);
  });

  it("preserves malformed and boolean union members at their original indexes", () => {
    const options = [null, false, true, 0, "object", undefined, {}];

    expect(
      stripEmptyProperties(options as unknown as StrictRJSFSchema[]),
    ).toEqual(options);
  });

  it.each(
    [null, false, true, 0, "invalid", [], ["invalid"]].map((properties) => ({
      properties,
    })),
  )(
    "leaves malformed properties $properties for the validator to reject",
    ({ properties }) => {
      const option = { type: "object", properties };
      const [result] = stripEmptyProperties([
        option as unknown as StrictRJSFSchema,
      ]);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      try {
        expect(result).toBe(option);
        expect(customValidator.isValid(result, {}, {})).toBe(false);
      } finally {
        warn.mockRestore();
      }
    },
  );

  it.each(["__proto__", "constructor", "prototype"])(
    "preserves %s as an own property without changing prototypes",
    (name) => {
      const before = Object.getOwnPropertyDescriptors(Object.prototype);
      const option: StrictRJSFSchema = JSON.parse(
        `{"type":"object","properties":{},"${name}":{"marker":true}}`,
      );
      const [result] = stripEmptyProperties([option]);
      expect(Object.hasOwn(result, name)).toBe(true);
      expect(Object.getOwnPropertyDescriptor(result, name)?.value).toEqual({
        marker: true,
      });
      expect(Object.getPrototypeOf(result)).toBe(Object.prototype);
      expect(
        getFirstMatchingOption(
          customValidator,
          {},
          [{ type: "string" }, result],
          {},
        ),
      ).toBe(1);

      const withProperty: StrictRJSFSchema = JSON.parse(
        `{"type":"object","properties":{"${name}":{"type":"string"}}}`,
      );
      expect(stripEmptyProperties([withProperty])[0]).toBe(withProperty);
      const data = JSON.parse(`{"${name}":"value"}`);
      expect(customValidator.isValid(withProperty, data, withProperty)).toBe(
        true,
      );
      expect(Object.getOwnPropertyDescriptors(Object.prototype)).toEqual(
        before,
      );
    },
  );
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
