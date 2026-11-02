export type JSONRecord = Record<string, unknown>;

export function asRecord(value: unknown): JSONRecord | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JSONRecord)
    : null;
}

export function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function parseJSONLoose(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}

export function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

const SECRET_KEY =
  /(pass(word)?|secret|token|api[_-]?key|authorization|cookie|credential|private[_-]?key|session[_-]?id)/i;

interface BoundedOptions {
  maxChars?: number;
  maxDepth?: number;
  maxItems?: number;
  maxString?: number;
}

/**
 * Pretty-prints untrusted tool data for a details view. Secret-looking keys
 * are masked and every dimension is capped, so a huge or hostile payload
 * stays readable and never leaks a credential into a screenshot.
 */
export function boundedPrettyPrint(
  value: unknown,
  {
    maxChars = 20_000,
    maxDepth = 6,
    maxItems = 60,
    maxString = 2_000,
  }: BoundedOptions = {},
): string {
  function walk(node: unknown, depth: number): unknown {
    if (typeof node === "string") return truncate(node, maxString);
    if (node === null || typeof node !== "object") return node;
    if (depth >= maxDepth) return Array.isArray(node) ? "[…]" : "{…}";
    if (Array.isArray(node)) {
      const items = node
        .slice(0, maxItems)
        .map((item) => walk(item, depth + 1));
      if (node.length > maxItems)
        items.push(`… ${node.length - maxItems} more`);
      return items;
    }
    const entries = Object.entries(node as JSONRecord);
    const result: JSONRecord = {};
    for (const [key, child] of entries.slice(0, maxItems)) {
      result[key] = SECRET_KEY.test(key) ? "••••••" : walk(child, depth + 1);
    }
    if (entries.length > maxItems) {
      result["…"] = `${entries.length - maxItems} more fields`;
    }
    return result;
  }

  const shaped = walk(value, 0);
  const text =
    typeof shaped === "string"
      ? shaped
      : (JSON.stringify(shaped, null, 2) ?? "");
  return truncate(text, maxChars);
}
