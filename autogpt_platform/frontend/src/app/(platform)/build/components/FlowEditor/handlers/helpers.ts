const DELIMITTER = "_#_";

/**
 * Convert RJSF id to handle id
 * Extracts the last meaningful part from an RJSF field ID
 */
export const fromRjsfId = (id: string): string => {
  if (!id) return "";
  const parts = id.split("_");
  const filtered = parts.filter(
    (p) => p !== "root" && p !== "properties" && p.length > 0,
  );
  return filtered.at(-1) || "";
};

/**
 * Compose a nested key for handles; consistent with legacy builder
 * Joins multiple parts with a delimiter after normalizing each part
 */
export const composeKey = (parts: string[]): string => {
  const cleaned = parts.filter(Boolean).map((p) => normalizeKey(p));
  return cleaned.join(DELIMITTER);
};

/**
 * Normalize a key by cleaning up whitespace and special characters
 * Converts to lowercase and replaces invalid characters with underscores
 */
export const normalizeKey = (key: string): string => {
  return key
    .trim()
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_\-]/g, "_")
    .toLowerCase();
};
