import { RJSFSchema, StrictRJSFSchema } from "@rjsf/utils";

const TYPE_PRIORITY = [
  "string",
  "number",
  "integer",
  "boolean",
  "array",
  "object",
] as const;

export function getDefaultTypeIndex(options: StrictRJSFSchema[]): number {
  for (const preferredType of TYPE_PRIORITY) {
    const index = options.findIndex((opt) => opt.type === preferredType);
    if (index >= 0) return index;
  }

  const nonNullIndex = options.findIndex((opt) => opt.type !== "null");
  return nonNullIndex >= 0 ? nonNullIndex : 0;
}

/**
 * Returns shallow copies of the options with empty `properties` objects removed.
 *
 * rjsf's `retrieveSchema` stubs `properties: {}` onto open-ended object schemas
 * (`additionalProperties: true`). `getFirstMatchingOption` then builds
 * `anyOf: []` from those keys, which AJV rejects as an invalid schema.
 * Order is preserved so returned indices still map to the original options.
 */
export function stripEmptyProperties<T extends StrictRJSFSchema>(
  options: T[],
): T[] {
  return options.map((option) => {
    const properties = option?.properties;
    if (
      properties &&
      typeof properties === "object" &&
      Object.keys(properties).length === 0
    ) {
      const copy = { ...option };
      delete copy.properties;
      return copy;
    }
    return option;
  });
}

/**
 * Determines if a type selector should be shown for an anyOf schema
 * Returns false for simple optional types (type | null)
 * Returns true for complex anyOf (3+ types or multiple non-null types)
 */
export function shouldShowTypeSelector(
  schema: RJSFSchema | undefined,
): boolean {
  const anyOf = schema?.anyOf;
  if (!anyOf || !Array.isArray(anyOf) || anyOf.length === 0) {
    return false;
  }

  if (anyOf.length === 2 && anyOf.some((opt: any) => opt.type === "null")) {
    return false;
  }

  return anyOf.length >= 3;
}

export function isSimpleOptional(schema: RJSFSchema | undefined): boolean {
  const anyOf = schema?.anyOf;
  return (
    Array.isArray(anyOf) &&
    anyOf.length === 2 &&
    anyOf.some((opt: any) => opt.type === "null")
  );
}

export function getOptionalType(
  schema: RJSFSchema | undefined,
): string | undefined {
  if (!isSimpleOptional(schema)) {
    return undefined;
  }

  const anyOf = schema?.anyOf;
  const nonNullOption = anyOf?.find((opt: any) => opt.type !== "null");
  return nonNullOption ? (nonNullOption as any).type : undefined;
}
