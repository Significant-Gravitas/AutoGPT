export function asObject(value: unknown): Record<string, unknown> | null {
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? parsed
        : null;
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

export function safeHostname(url: string): string | null {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
}

export function integrationIconSrc(provider: string): string | null {
  const slug = provider
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_")
    .replace(/[^a-z0-9_]/g, "");
  return slug ? `/integrations/${slug}.png` : null;
}

// Every backend tool response carries these envelope fields; cards read the
// domain payload, so shape detection runs on the stripped object.
const BASE_RESPONSE_KEYS = new Set(["type", "message", "session_id"]);

export function stripBaseFields(
  obj: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(obj).filter(([key]) => !BASE_RESPONSE_KEYS.has(key)),
  );
}

// Block/agent outputs arrive as {output_name: [values]}; flatten to
// {name, value} entries for OutputList.
export function dictToOutputItems(
  value: unknown,
): Record<string, unknown>[] | null {
  const obj =
    value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;
  if (!obj) return null;
  const entries = Object.entries(obj);
  if (entries.length === 0) return null;
  return entries.map(([name, entryValue]) => ({
    name,
    value:
      Array.isArray(entryValue) && entryValue.length === 1
        ? entryValue[0]
        : entryValue,
  }));
}

export function asItems(value: unknown): Record<string, unknown>[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  return value.map((item) =>
    item && typeof item === "object"
      ? (item as Record<string, unknown>)
      : { value: item },
  );
}

export function str(
  obj: Record<string, unknown>,
  ...keys: string[]
): string | null {
  for (const key of keys) {
    const value = obj[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return null;
}

export function inline(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean")
    return String(value);
  const json = JSON.stringify(value);
  return json && json.length > 120 ? json.slice(0, 120) + "…" : (json ?? "");
}

export function resultItemKey(
  item: Record<string, unknown>,
  fallbackIndex: number,
): string {
  for (const field of [
    "id",
    "step_id",
    "keyword",
    "execution_id",
    "identifier",
    "path",
    "url",
    "name",
    "title",
    "content",
  ]) {
    const value = item[field];
    if (typeof value === "string" || typeof value === "number") {
      return `${field}:${value}`;
    }
  }
  return `item:${fallbackIndex}:${inline(item)}`;
}

export function humanizeKey(key: string): string {
  return key.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

const DATE_TIME_FORMATTER = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

export function formatWhen(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return DATE_TIME_FORMATTER.format(date);
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}
