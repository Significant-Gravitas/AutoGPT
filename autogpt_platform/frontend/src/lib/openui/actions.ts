export function getActionFields(state: unknown, formName?: string) {
  if (!state || typeof state !== "object") return {};
  const record = state as Record<string, unknown>;
  const fields = formName ? record[formName] : record;
  if (!fields || typeof fields !== "object") return {};
  const result: Record<string, string | number | boolean> = {};
  for (const [name, field] of Object.entries(fields)) {
    const value =
      field && typeof field === "object" && "value" in field
        ? field.value
        : field;
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    )
      result[name] = value;
  }
  return result;
}
