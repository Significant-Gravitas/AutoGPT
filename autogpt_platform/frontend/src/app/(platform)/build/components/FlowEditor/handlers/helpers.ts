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
  return filtered_id;
};

export const generateHandleIdFromTitleId = (
  fieldKey: string,
  {
    isObjectProperty,
    isAdditionalProperty,
    isArrayItem,
  }: {
    isArrayItem?: boolean;
    isObjectProperty?: boolean;
    isAdditionalProperty?: boolean;
  } = {
    isArrayItem: false,
    isObjectProperty: false,
    isAdditionalProperty: false,
  },
): string => {
  if (!fieldKey) return "";

  const filteredKey = cleanTitleId(fieldKey);
  if (isAdditionalProperty || isArrayItem) {
    return filteredKey;
  }
  const cleanedKey = sanitizeForHandleId(filteredKey);

  if (isObjectProperty) {
    // "config_api_key" -> "config.api_key"
    const parts = cleanedKey.split("_");
    if (parts.length >= 2) {
      const baseName = parts[0];
      const propertyName = parts.slice(1).join("_");
      return `${baseName}.${propertyName}`;
    }
  }

  return cleanedKey;
};
