import type { RJSFSchema } from "@rjsf/utils";
import { customizeValidator } from "@rjsf/validator-ajv8";

export function buildInputSchema(inputs: unknown): RJSFSchema | null {
  if (!inputs || typeof inputs !== "object") return null;
  const properties = inputs as RJSFSchema["properties"];
  if (!properties || Object.keys(properties).length === 0) return null;
  return inputs as RJSFSchema;
}

export function extractDefaults(schema: RJSFSchema): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  const props = schema.properties;
  if (!props || typeof props !== "object") return defaults;

  for (const [key, prop] of Object.entries(props)) {
    if (typeof prop !== "object" || prop === null) continue;
    if ("default" in prop && prop.default !== undefined) {
      defaults[key] = prop.default;
    } else if (
      "examples" in prop &&
      Array.isArray(prop.examples) &&
      prop.examples.length > 0
    ) {
      defaults[key] = prop.examples[0];
    }
  }
  return defaults;
}

export function isFormValid(
  schema: RJSFSchema,
  formData: Record<string, unknown>,
): boolean {
  const { errors } = integerTextValidator.validateFormData(
    formData,
    withIntegerText(schema),
  );
  return errors.length === 0;
}

// Integer inputs past Number.MAX_SAFE_INTEGER hold digit strings
// (see parseIntegerInput), so integer schemas must accept them, within the
// same bounds a number is held to.
const INTEGER_TEXT = "x-integer-text";
const DATA_KEYWORDS = new Set(["default", "const", "enum", "examples"]);
const SCHEMA_MAPS = new Set([
  "properties",
  "patternProperties",
  "dependencies",
  "dependentSchemas",
  "$defs",
  "definitions",
]);
const widenedSchemas = new WeakMap<RJSFSchema, RJSFSchema>();

// Validates like the shared customValidator (no format checks, unknown
// keywords allowed), plus the digit-string keyword.
const integerTextValidator = customizeValidator({
  ajvOptionsOverrides: { strict: false, validateFormats: false },
  extenderFn: (ajv) =>
    ajv.addKeyword({
      keyword: INTEGER_TEXT,
      type: "string",
      validate: (_: unknown, text: string, fieldSchema?: object) =>
        isIntegerWithin(text, fieldSchema ?? {}),
    }),
});

interface IntegerBounds {
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  multipleOf?: number;
}

// A bigint compares with a number by exact value, where Number(text) would
// round past 2^53.
function isIntegerWithin(text: string, bounds: IntegerBounds): boolean {
  if (!/^-?\d+$/.test(text)) return false;
  const n = BigInt(text);
  const { minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf } =
    bounds;
  return (
    (minimum === undefined || n >= minimum) &&
    (maximum === undefined || n <= maximum) &&
    (exclusiveMinimum === undefined || n > exclusiveMinimum) &&
    (exclusiveMaximum === undefined || n < exclusiveMaximum) &&
    (multipleOf === undefined ||
      !Number.isInteger(multipleOf) ||
      n % BigInt(multipleOf) === 0n)
  );
}

function withIntegerText(schema: RJSFSchema): RJSFSchema {
  let widened = widenedSchemas.get(schema);
  if (!widened) {
    widened = widenSchema(schema) as RJSFSchema;
    widenedSchemas.set(schema, widened);
  }
  return widened;
}

function widenSchema(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(widenSchema);
  if (!isRecord(node)) return node;
  const out = Object.fromEntries(
    Object.entries(node).map(([key, value]) => [key, widenKeyword(key, value)]),
  );
  const types = [out.type].flat();
  if (!types.includes("integer") || types.includes("string")) return out;
  return { ...out, type: [...types, "string"], [INTEGER_TEXT]: true };
}

function widenKeyword(key: string, value: unknown): unknown {
  if (DATA_KEYWORDS.has(key)) return value;
  if (key === "oneOf" && Array.isArray(value)) return widenOneOf(value);
  if (SCHEMA_MAPS.has(key) && isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([name, schema]) => [
        name,
        widenSchema(schema),
      ]),
    );
  }
  return widenSchema(value);
}

// The widened oneOf has one if/then/else branch. A string that a branch
// accepts as text keeps the oneOf result it had, so a widened integer branch
// can't make it match twice; only for a digit string no branch accepts do the
// widened branches decide, still as a oneOf.
function widenOneOf(branches: unknown[]): unknown[] {
  return [
    {
      if: { type: "string", anyOf: branches },
      then: { oneOf: branches },
      else: { oneOf: branches.map(widenSchema) },
    },
  ];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
