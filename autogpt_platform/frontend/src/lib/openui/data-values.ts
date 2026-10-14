import { z } from "zod/v4";
import { asNumber } from "./calculations";

export const identifier = z.string().regex(/^[a-zA-Z][a-zA-Z0-9_-]{0,79}$/);
export const shortText = z.string().min(1).max(200);
export const sourceLink = z.object({
  label: shortText,
  url: z.url({ protocol: /^https?$/ }).max(2000),
});

export function compareCells(a: unknown, b: unknown) {
  if (a == null || a === "") return b == null || b === "" ? 0 : 1;
  if (b == null || b === "") return -1;
  const left = asNumber(a);
  const right = asNumber(b);
  if (left !== null && right !== null) return left - right;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}
