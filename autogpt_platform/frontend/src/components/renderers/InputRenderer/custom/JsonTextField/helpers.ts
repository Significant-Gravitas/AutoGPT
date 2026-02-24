import { RJSFSchema } from "@rjsf/utils";

/**
 * Converts form data to a JSON string for display
 * @param formData - The data to stringify
 * @returns JSON string or empty string if data is null/undefined
 */
export function stringifyFormData(formData: unknown): string {
  if (formData === undefined || formData === null) {
    return "";
  }
  try {
    return JSON.stringify(formData, null, 2);
  } catch {
    return "";
  }
}

/**
 * Parses a JSON string into an object/array
 * @param value - The JSON string to parse
 * @returns Parsed value or undefined if parsing fails or empty
 */
export function parseJsonValue(value: string): unknown | undefined {
  const trimmed = value.trim();
  if (trimmed === "") {
    return undefined;
  }

  try {
    return JSON.parse(trimmed);
  } catch {
    return undefined;
  }
}

/**
 * Gets the appropriate placeholder text based on schema type
 * @param schema - The JSON schema
 * @returns Placeholder string
 */
export function getPlaceholder(schema: RJSFSchema): string {
  if (schema.type === "array") {
    return '["item1", "item2"] or [{"key": "value"}]';
  }
  if (schema.type === "object") {
    return '{"key": "value"}';
  }
  return "Enter JSON value...";
}

/**
 * Checks if a JSON string is valid
 * @param value - The JSON string to validate
 * @returns true if valid JSON, false otherwise
 */
export function isValidJson(value: string): boolean {
  if (value.trim() === "") {
    return true; // Empty is considered valid (will be undefined)
  }
  try {
    JSON.parse(value);
    return true;
  } catch {
    return false;
  }
}
