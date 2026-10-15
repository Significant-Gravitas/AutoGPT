import type { ToolPart } from "./presentation";

const MAX_ENTRIES = 100;
const parts = new Map<string, ToolPart>();

/**
 * Hands a tool part to the details sheet without putting tool data in a
 * route URL. Entries are memory-only and cleared on sign-out.
 */
export function rememberToolPart(part: ToolPart): string {
  const key = part.toolCallID;
  parts.delete(key);
  parts.set(key, part);
  while (parts.size > MAX_ENTRIES) {
    const oldest = parts.keys().next().value;
    if (oldest === undefined) break;
    parts.delete(oldest);
  }
  return key;
}

export function recallToolPart(key: string): ToolPart | null {
  return parts.get(key) ?? null;
}

export function forgetToolParts() {
  parts.clear();
}
