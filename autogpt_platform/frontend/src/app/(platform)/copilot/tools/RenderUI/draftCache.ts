import { record } from "@/lib/openui/draft-fields";

interface CachedDraft {
  source: string;
  state: Record<string, unknown>;
  revision: number;
  dirty?: boolean;
}
export function readDraftCache(key: string | null): CachedDraft | null {
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw || raw.length > 320_000) return null;
    const value = record(JSON.parse(raw));
    return value &&
      typeof value.source === "string" &&
      typeof value.revision === "number" &&
      record(value.state)
      ? (value as unknown as CachedDraft)
      : null;
  } catch {
    return null;
  }
}
export function writeDraftCache(key: string | null, draft: CachedDraft) {
  if (!key) return;
  try {
    const text = JSON.stringify(draft);
    if (text.length <= 320_000) localStorage.setItem(key, text);
  } catch {}
}
