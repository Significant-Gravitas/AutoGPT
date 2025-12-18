/**
 * Handle ID Types for different input structures
 *
 * Examples:
 * SIMPLE: "message"
 * NESTED: "config.api_key"
 * ARRAY: "items_$_0", "items_$_1"
 * KEY_VALUE: "headers_#_Authorization", "params_#_limit"
 *
 * Note: All handle IDs are sanitized to remove spaces and special characters.
 * Spaces become underscores, and special characters are removed.
 * Example: "user name" becomes "user_name", "email@domain.com" becomes "emaildomaincom"
 */
export enum HandleIdType {
  SIMPLE = "SIMPLE",
  NESTED = "NESTED",
  ARRAY = "ARRAY",
  KEY_VALUE = "KEY_VALUE",
}

const fromRjsfId = (id: string): string => {
  if (!id) return "";
  const parts = id.split("_");
  const filtered = parts.filter(
    (p) => p !== "root" && p !== "properties" && p.length > 0,
  );
  return filtered.join("_") || "";
};

const sanitizeForHandleId = (str: string): string => {
  if (!str) return "";

  return str
    .trim()
    .replace(/\s+/g, "_") // Replace spaces with underscores
    .replace(/[^a-zA-Z0-9_-]/g, "") // Remove special characters except underscores and hyphens
    .replace(/_+/g, "_") // Replace multiple consecutive underscores with single underscore
    .replace(/^_|_$/g, ""); // Remove leading/trailing underscores
};

export const generateHandleId = (
  fieldKey: string,
  nestedValues: string[] = [],
  type: HandleIdType = HandleIdType.SIMPLE,
): string => {
  if (!fieldKey) return "";

  fieldKey = fromRjsfId(fieldKey);
  fieldKey = sanitizeForHandleId(fieldKey);

  if (type === HandleIdType.SIMPLE || nestedValues.length === 0) {
    return fieldKey;
  }

  const sanitizedNestedValues = nestedValues.map((value) =>
    sanitizeForHandleId(value),
  );

  switch (type) {
    case HandleIdType.NESTED:
      return [fieldKey, ...sanitizedNestedValues].join(".");

    case HandleIdType.ARRAY:
      return [fieldKey, ...sanitizedNestedValues].join("_$_");

    case HandleIdType.KEY_VALUE:
      return [fieldKey, ...sanitizedNestedValues].join("_#_");

    default:
      return fieldKey;
  }
};

export const parseKeyValueHandleId = (
  handleId: string,
  type: HandleIdType,
): string => {
  if (type === HandleIdType.KEY_VALUE) {
    return handleId.split("_#_")[1];
  } else if (type === HandleIdType.ARRAY) {
    return handleId.split("_$_")[1];
  } else if (type === HandleIdType.NESTED) {
    return handleId.split(".")[1];
  } else if (type === HandleIdType.SIMPLE) {
    return handleId.split("_")[1];
  }
  return "";
};
