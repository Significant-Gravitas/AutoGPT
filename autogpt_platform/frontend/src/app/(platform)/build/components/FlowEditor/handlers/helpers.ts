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
  mainKey: string,
  nestedValues: string[] = [],
  type: HandleIdType = HandleIdType.SIMPLE,
): string => {
  if (!mainKey) return "";

  mainKey = fromRjsfId(mainKey);
  mainKey = sanitizeForHandleId(mainKey);

  if (type === HandleIdType.SIMPLE || nestedValues.length === 0) {
    return mainKey;
  }

  const sanitizedNestedValues = nestedValues.map((value) =>
    sanitizeForHandleId(value),
  );

  switch (type) {
    case HandleIdType.NESTED:
      return [mainKey, ...sanitizedNestedValues].join(".");

    case HandleIdType.ARRAY:
      return [mainKey, ...sanitizedNestedValues].join("_$_");

    case HandleIdType.KEY_VALUE:
      return [mainKey, ...sanitizedNestedValues].join("_#_");

    default:
      return mainKey;
  }
};

export const parseHandleId = (
  handleId: string,
): {
  mainKey: string;
  nestedValues: string[];
  type: HandleIdType;
} => {
  if (!handleId) {
    return { mainKey: "", nestedValues: [], type: HandleIdType.SIMPLE };
  }

  if (handleId.includes("_#_")) {
    const parts = handleId.split("_#_");
    return {
      mainKey: parts[0],
      nestedValues: parts.slice(1),
      type: HandleIdType.KEY_VALUE,
    };
  }

  if (handleId.includes("_$_")) {
    const parts = handleId.split("_$_");
    return {
      mainKey: parts[0],
      nestedValues: parts.slice(1),
      type: HandleIdType.ARRAY,
    };
  }

  if (handleId.includes(".")) {
    const parts = handleId.split(".");
    return {
      mainKey: parts[0],
      nestedValues: parts.slice(1),
      type: HandleIdType.NESTED,
    };
  }

  return {
    mainKey: handleId,
    nestedValues: [],
    type: HandleIdType.SIMPLE,
  };
};
