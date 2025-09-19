/**
 * Handle ID Types for different input structures
 *
 * Examples:
 * SIMPLE: "message"
 * NESTED: "config.api_key"
 * ARRAY: "items_$_0", "items_$_1"
 * KEY_VALUE: "headers_#_Authorization", "params_#_limit"
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

export const generateHandleId = (
  mainKey: string,
  nestedValues: string[] = [],
  type: HandleIdType = HandleIdType.SIMPLE,
): string => {
  if (!mainKey) return "";

  mainKey = fromRjsfId(mainKey);

  if (type === HandleIdType.SIMPLE || nestedValues.length === 0) {
    return mainKey;
  }

  switch (type) {
    case HandleIdType.NESTED:
      return [mainKey, ...nestedValues].join(".");

    case HandleIdType.ARRAY:
      return [mainKey, ...nestedValues].join("_$_");

    case HandleIdType.KEY_VALUE:
      return [mainKey, ...nestedValues].join("_#_");

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
