export { integrationIconSrc } from "@/components/molecules/IntegrationLogo/helpers";

/** Text the backend escaped for a model's context (`&`, `<`, `>`), as the
 *  user reads it. `&amp;` is decoded last, so a literal `&lt;` survives. */
export function unescapePromptTags(value: string): string {
  return value
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

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

// Anyone holding a webhook ingress URL can trigger the agent unless the
// trigger sets a secret_token, so it is treated as a credential.
export function isWebhookIngressUrl(url: string): boolean {
  return /\/webhooks\/[^/?#]+\/ingress(?:[/?#]|$)/.test(url);
}

export function safeHostname(url: string): string | null {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
}

// Every backend tool response carries these envelope fields; cards read the
// domain payload, so shape detection runs on the stripped object.
const BASE_RESPONSE_KEYS = new Set(["type", "message", "session_id"]);

export function stripBaseFields(
  obj: Record<string, unknown>,
): Record<string, unknown> {
  // A couple of responses repeat their own discriminator in `name`
  // ("no_results", "agents_found"). It is envelope, not payload, and left in
  // it both renders as a field of its own and pushes the object past the
  // single-key check that picks a real card -- so a no-results answer came out
  // as a raw suggestions array labelled "Suggestions", with "Name no_results"
  // under it. Only drop `name` when it is that echo: a payload whose name
  // means something (a folder, a file, a block) never equals the type.
  const type = obj.type;
  return Object.fromEntries(
    Object.entries(obj).filter(
      ([key, value]) =>
        !BASE_RESPONSE_KEYS.has(key) && !(key === "name" && value === type),
    ),
  );
}

// What a lookup was scoped to (an expert, a chat platform), echoed back.
const SCOPE_KEY = /(^|_)id$|^platform$/;
// Where a page starts and how long it is, not how much there is.
const PAGING_KEYS = new Set([
  "page",
  "page_size",
  "per_page",
  "limit",
  "offset",
]);
// How many there are: zero is "none", any other number is an answer.
const COUNT_KEY = /(^|_)count$|^total$/;
// Set when there is a further page; false says this one is all of it.
const MORE_KEYS = new Set(["has_more", "has_next"]);

function isEmptyCollection(value: unknown): boolean {
  if (Array.isArray(value)) return value.length === 0;
  return (
    typeof value === "object" &&
    value !== null &&
    Object.keys(value).length === 0
  );
}

function isBookkeeping(key: string, value: unknown): boolean {
  if (isEmptyCollection(value) || value === "") return true;
  if (typeof value === "number")
    return PAGING_KEYS.has(key) || (value === 0 && COUNT_KEY.test(key));
  if (typeof value === "string") return SCOPE_KEY.test(key);
  return value === false && MORE_KEYS.has(key);
}

/** A lookup that came back with nothing: every list in it is empty, and the
 *  rest is bookkeeping (what it was scoped to, zero counts, paging). A result
 *  that names or states anything (`name`, `ok`), or holds more than it listed
 *  (`total_count: 5`, `has_more: true`), is an answer, even next to an empty
 *  list. */
export function isEmptyResult(output: unknown): boolean {
  const obj = asObject(output);
  if (!obj) return false;
  const entries = Object.entries(stripBaseFields(obj)).filter(
    ([, value]) => value !== null && value !== undefined,
  );
  if (!entries.some(([, value]) => Array.isArray(value))) return false;
  return entries.every(([key, value]) => isBookkeeping(key, value));
}

/** How an empty result reads wherever one is shown. */
export const NOTHING_FOUND = "Nothing found.";

/** Output with nothing in it at all: `null`, an empty list or object, blank
 *  text, or that same JSON as text (an MCP tool's empty search answers "[]").
 *  `undefined` is a call still running, not an empty answer. */
export function isBlankOutput(value: unknown): boolean {
  if (typeof value !== "string")
    return value === null || isEmptyCollection(value);
  const text = value.trim();
  if (!text) return true;
  if (!/^(\[|\{|null$)/.test(text)) return false;
  try {
    const parsed: unknown = JSON.parse(text);
    return parsed === null || isEmptyCollection(parsed);
  } catch {
    return false;
  }
}

/** A value as a result card shows it: empty lists and objects read "None"
 *  instead of their JSON. */
export function displayValue(value: unknown): string {
  return isEmptyCollection(value) ? "None" : inline(value);
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

interface TruncatedOutput {
  totalChars: number;
  preview: string;
}

const TRUNCATED_RE =
  /^<tool-output-truncated total_chars=(\d+)[^>]*>\n([\s\S]*)\n<\/tool-output-truncated>$/;

/** Unwrap the `<tool-output-truncated>` envelope the backend wraps oversized
 *  tool output in, dropping the read_workspace_file instructions meant for the
 *  model rather than the reader. */
export function parseTruncatedOutput(value: string): TruncatedOutput | null {
  const match = TRUNCATED_RE.exec(value.trim());
  if (!match) return null;
  return {
    totalChars: Number(match[1]),
    preview: match[2].split("\nFull output (")[0].trim(),
  };
}
