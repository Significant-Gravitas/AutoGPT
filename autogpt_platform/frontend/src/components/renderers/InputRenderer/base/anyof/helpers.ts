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
