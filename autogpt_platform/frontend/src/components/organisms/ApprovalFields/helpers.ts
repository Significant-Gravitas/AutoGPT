import { formatDistanceToNow } from "date-fns";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";

export const REDACTED = "[redacted]";
export const MAX_FIELDS = 6;
export const CLAMP_LINES = 3;
export const CLAMP_CHARS = 280;
export const CODE_MAX_LINES = 12;
export const ARRAY_SHOWN = 3;
export const ARRAY_MAX = 5;

// Arguments whose value is the content, shown as code rather than prose.
const CODE_KEYS = new Set(["command", "code", "script", "source", "sql"]);

export type FieldKind =
  | "secret"
  | "code"
  | "long"
  | "short"
  | "list"
  | "object"
  | "json";

export interface FieldSpec {
  key: string;
  label: string;
}

// What an id argument names, resolved by the server when the call was held.
export interface Reference {
  key: string;
  entity: string;
  id: string;
  name: string | null;
  href: string | null;
  // The hover card: the thing's family, its own prose, and short facts.
  kind: string | null;
  description: string | null;
  meta: Fact[];
  avatarURL: string | null;
  avatarColor: string | null;
  skills: string[];
  summary: string | null;
}

// A card fact: `text` as stored, or a cadence or time the card words for the viewer.
export interface Fact {
  text: string;
  cron: string | null;
  label: string | null;
  at: string | null;
}

export function factText(fact: Fact) {
  if (fact.cron) return cronText(fact.cron) ?? fact.text;
  if (fact.label && fact.at) {
    const at = new Date(fact.at);
    if (!Number.isNaN(at.getTime()))
      return `${fact.label} ${formatDistanceToNow(at, { addSuffix: true })}`;
  }
  return fact.text;
}

function cronText(cron: string) {
  try {
    return humanizeCronExpression(cron);
  } catch {
    return null;
  }
}

export function fieldKind(key: string, value: unknown): FieldKind {
  if (value === REDACTED) return "secret";
  if (Array.isArray(value)) {
    return value.every(isScalar) ? "list" : "json";
  }
  if (value && typeof value === "object") {
    return Object.values(value).every(
      (v) => isScalar(v) || (Array.isArray(v) && v.every(isScalar)),
    )
      ? "object"
      : "json";
  }
  const text = String(value);
  if (CODE_KEYS.has(key) || key.endsWith("_code")) return "code";
  if (text.length > CLAMP_CHARS || lineCount(text) > CLAMP_LINES) return "long";
  return "short";
}

interface VisibleKeysArgs {
  keys: string[];
  values: Record<string, unknown>;
  hiddenKeys: string[];
  // Show ids when nothing else would tell this call from another.
  idsWhenAlone: boolean;
  references?: Reference[];
}

export function visibleKeys({
  keys,
  values,
  hiddenKeys,
  idsWhenAlone,
  references = [],
}: VisibleKeysArgs) {
  const present = [...new Set(keys)].filter(
    (key) => !hiddenKeys.includes(key) && hasValue(values[key]),
  );
  // An id argument the server looked up is shown whatever it found: its names,
  // or the raw ids when none resolved, even beside a named headline.
  const referenced = new Set(references.map((ref) => ref.key));
  const named = present.filter((key) => !isIdKey(key) || referenced.has(key));
  return named.length > 0 || !idsWhenAlone ? named : present;
}

export function isIdKey(key: string) {
  return /(^|_)ids?$/.test(key);
}

function hasValue(value: unknown) {
  if (value === null || value === undefined || value === "") return false;
  if (value === false) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
}

export function listText(values: unknown[]) {
  if (values.length <= ARRAY_MAX) return values.map(String).join(", ");
  return `${values.slice(0, ARRAY_SHOWN).map(String).join(", ")} +${values.length - ARRAY_SHOWN} more`;
}

export function scalarText(value: unknown) {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

export function lineCount(text: string) {
  return text.split("\n").length;
}

export function humanize(key: string) {
  const words = key
    .replace(/_/g, " ")
    .replace(
      /([a-z])([A-Z])/g,
      (_, a: string, b: string) => `${a} ${b.toLowerCase()}`,
    )
    .trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function isScalar(value: unknown) {
  return (
    value === null || ["string", "number", "boolean"].includes(typeof value)
  );
}
