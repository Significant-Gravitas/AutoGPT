// Here we are handling single level of nesting, if need more in future then i will update it

const sanitizeForHandleId = (str: string): string => {
  if (!str) return "";

  return str
    .trim()
    .replace(/\s+/g, "_") // Replace spaces with underscores
    .replace(/[^a-zA-Z0-9_-]/g, "") // Remove special characters except underscores and hyphens
    .replace(/_+/g, "_") // Replace multiple consecutive underscores with single underscore
    .replace(/^_|_$/g, ""); // Remove leading/trailing underscores
};

const cleanTitleId = (id: string): string => {
  if (!id) return "";

  if (id.endsWith("_title")) {
    id = id.slice(0, -6);
  }
  const parts = id.split("_");
  const filtered = parts.filter(
    (p) => p !== "root" && p !== "properties" && p.length > 0,
  );
  const filtered_id = filtered.join("_") || "";
  return sanitizeForHandleId(filtered_id);
};

export const generateHandleIdFromTitleId = (
  fieldKey: string,
  {
    isArrayItem,
    isObjectProperty,
  }: { isArrayItem: boolean; isObjectProperty: boolean } = {
    isArrayItem: false,
    isObjectProperty: false,
  },
): string => {
  if (!fieldKey) return "";

  const cleanedKey = cleanTitleId(fieldKey);

  if (isArrayItem) {
    // For array items: convert "items_0" to "items_$_0"
    // Find the last underscore followed by a number and replace with _$_
    const match = cleanedKey.match(/^(.+)_(\d+)$/);
    if (match) {
      return `${match[1]}_$_${match[2]}`;
    }
    return cleanedKey;
  }

  if (isObjectProperty) {
    // For object properties: convert "config_api_key" to "config_#_api_key"
    // or handle nested dot notation
    // will see it
    const parts = cleanedKey.split("_");
    if (parts.length >= 2) {
      const baseName = parts[0];
      const propertyName = parts.slice(1).join("_");
      return `${baseName}_#_${propertyName}`;
    }
    return cleanedKey;
  }

  // For simple fields or nested fields (dot notation handled elsewhere)
  return cleanedKey;
};
