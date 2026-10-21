import { describe, expect, it } from "vitest";
import { RJSFSchema } from "@rjsf/utils";
import { generateUiSchemaForCustomFields } from "../utils/generate-ui-schema";
import { JSON_TEXT_FIELD_ID } from "../custom/custom-registry";

function arrayOf(items: RJSFSchema): RJSFSchema {
  return {
    type: "object",
    properties: {
      filters: { type: "array", title: "Filters", items },
    },
  };
}

describe("generateUiSchemaForCustomFields — array items", () => {
  it("renders items with named properties as a sub-form, not a JSON textarea", () => {
    // A block input like `filters: list[FilterCondition]`, where the condition
    // has an enum column. Rendering this as JSON forces users to hand-type
    // values from a 40+ member enum.
    const ui = generateUiSchemaForCustomFields(
      arrayOf({
        type: "object",
        title: "FilterCondition",
        properties: {
          column: { type: "string", enum: ["current_title", "first_name"] },
          operator: { type: "string", enum: ["=", "like"] },
          value: { type: "string" },
        },
      }),
    );

    expect((ui.filters as any)?.items?.["ui:field"]).not.toBe(
      JSON_TEXT_FIELD_ID,
    );
  });

  it("still falls back to a JSON textarea for free-form dict items", () => {
    const ui = generateUiSchemaForCustomFields(
      arrayOf({ type: "object", additionalProperties: true }),
    );

    expect((ui.filters as any).items["ui:field"]).toBe(JSON_TEXT_FIELD_ID);
  });

  it("still falls back to a JSON textarea for nested array items", () => {
    const ui = generateUiSchemaForCustomFields(
      arrayOf({ type: "array", items: { type: "string" } }),
    );

    expect((ui.filters as any).items["ui:field"]).toBe(JSON_TEXT_FIELD_ID);
  });

  it("leaves arrays of primitives alone", () => {
    const ui = generateUiSchemaForCustomFields(
      arrayOf({ type: "string", enum: ["markdown", "html"] }),
    );

    expect((ui.filters as any)?.items?.["ui:field"]).toBeUndefined();
  });

  it("keeps routing item-level custom fields ahead of the object check", () => {
    const ui = generateUiSchemaForCustomFields(
      arrayOf({
        type: "object",
        credentials_provider: ["openai"],
        properties: { id: { type: "string" } },
      } as RJSFSchema),
    );

    expect((ui.filters as any).items["ui:field"]).toBe(
      "custom/credential_field",
    );
  });
});
