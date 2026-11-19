const PIN_PART = /_[$#@]_/;
const TITLE_KEYS = ["title", "name", "subject", "headline", "label"];
const LINK_KEYS = ["link", "url", "href", "html_url"];
const ISO_DATE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/;

export function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

export function isEmpty(value: unknown) {
  return (
    value === null ||
    value === undefined ||
    value === "" ||
    (Array.isArray(value) && value.length === 0) ||
    (isRecord(value) && Object.keys(value).length === 0)
  );
}

/** A field name as people read it: "pub_date" is "Pub date", and a part of
 *  a list pin such as "lists_$_0" is "Lists 1". */
export function labelOf(key: string) {
  const [base, part] = key.split(PIN_PART);
  const words = base
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim()
    .toLowerCase()
    .replace(/\b(agent|graph)(s?)\b/g, "workflow$2");
  const label = words ? words[0].toUpperCase() + words.slice(1) : key;
  if (part === undefined) return label;
  return /^\d+$/.test(part)
    ? `${label} ${Number(part) + 1}`
    : `${label} · ${part}`;
}

/** The pins worth showing. A block that takes a list gets it both whole and
 *  in parts ("lists" and "lists_$_0"); the parts are left out. */
export function visiblePins(value: Record<string, unknown> | null) {
  const entries = Object.entries(value ?? {});
  const whole = new Set(entries.map(([key]) => key));
  return entries.filter(([key]) => {
    const [base, part] = key.split(PIN_PART);
    return part === undefined || !whole.has(base);
  });
}

/** Outputs arrive as lists; a list of one shows its one item. */
export function unwrap(value: unknown) {
  return Array.isArray(value) && value.length === 1 ? value[0] : value;
}

export function isLink(value: unknown): value is string {
  return typeof value === "string" && /^https?:\/\/\S+$/.test(value.trim());
}

export function hostOf(link: string) {
  try {
    return new URL(link).hostname.replace(/^www\./, "");
  } catch {
    return link;
  }
}

/** Text with any markup taken off, as one readable run. */
export function plainText(value: string) {
  if (!/<[a-z][^>]*>/i.test(value)) return value;
  return value
    .replace(/<\/(p|div|li|h\d)>|<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

/** A date and time as people write it, or null when the text is not one. */
export function dateOf(value: string) {
  if (!ISO_DATE.test(value)) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function firstText(item: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = item[key];
    if (typeof value === "string" && value.trim()) return { key, value };
  }
  return null;
}

/** What names an item in a list, and where it points. */
export function headOf(item: Record<string, unknown>) {
  const title = firstText(item, TITLE_KEYS);
  const link = firstText(item, LINK_KEYS);
  return {
    title: title?.value ?? null,
    titleKey: title?.key ?? null,
    link: link && isLink(link.value) ? link.value : null,
    linkKey: link?.key ?? null,
  };
}

export function countLabel(count: number, one: string, many: string) {
  return `${count} ${count === 1 ? one : many}`;
}
