import type { StrictRJSFSchema } from "@rjsf/utils";
import { describe, expect, it } from "vitest";
import { customValidator } from "../../../utils/custom-validator";
import { stripEmptyProperties } from "../helpers";

const constraints: StrictRJSFSchema[] = [
  { additionalProperties: true },
  { additionalProperties: false },
  { additionalProperties: { type: "string" } },
  { required: ["name"] },
  { minProperties: 1, maxProperties: 2 },
  {
    patternProperties: { "^name": { type: "string" } },
    additionalProperties: false,
  },
  { propertyNames: { pattern: "^name" } },
  { dependencies: { name: ["other"] } },
  { allOf: [{ required: ["name"] }, { maxProperties: 1 }] },
  { anyOf: [{ required: ["name"] }, { required: ["other"] }] },
  { not: { required: ["name"] } },
  { if: { required: ["name"] }, then: { required: ["other"] } },
];
const values = [
  null,
  "text",
  1,
  [],
  {},
  { name: "value" },
  { name: 3 },
  { name: "value", other: true },
];

describe("anyOf matcher normalization with the production Ajv validator", () => {
  it.each(constraints)("retains validation constraints %j", (constraint) => {
    const schema: StrictRJSFSchema = {
      type: "object",
      properties: {},
      ...constraint,
    };
    const original = structuredClone(schema);
    const [normalized] = stripEmptyProperties([schema]);

    for (const value of values) {
      expect(customValidator.isValid(normalized, value, schema)).toBe(
        customValidator.isValid(schema, value, schema),
      );
    }
    expect(schema).toEqual(original);
  });

  it("retains root reference resolution", () => {
    const root: StrictRJSFSchema = {
      definitions: { value: { type: "string" } },
    };
    const schema: StrictRJSFSchema = {
      type: "object",
      properties: {},
      additionalProperties: { $ref: "#/definitions/value" },
    };
    const [normalized] = stripEmptyProperties([schema]);
    expect(customValidator.isValid(normalized, { name: "valid" }, root)).toBe(
      true,
    );
    expect(customValidator.isValid(normalized, { name: 1 }, root)).toBe(false);
    expect(schema.properties).toEqual({});
  });
});
