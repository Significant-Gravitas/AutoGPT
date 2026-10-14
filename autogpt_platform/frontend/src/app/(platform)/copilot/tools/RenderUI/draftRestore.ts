import { mergeDraft } from "@/lib/openui/draft-merge";
import type { CachedDraft } from "./draftCache";
import type { SavedDraft } from "./draftTargets";

export const SUPERSEDED_NOTICE =
  "A newer version of this view is available below. This older version is read-only.";

export interface DraftSnapshot {
  state: Record<string, unknown>;
  revision: number;
  writable: boolean;
  notice: string;
  key: number;
}

interface Args {
  saved: SavedDraft | null;
  cached: CachedDraft | null;
  source: string;
  key: number;
}

export function restoreDraft({ saved, cached, source, key }: Args) {
  const useCache =
    cached &&
    (!saved ||
      (cached.source === source &&
        saved.writable &&
        (cached.dirty !== false || cached.revision >= saved.revision)));
  const base = useCache ? cached : saved;
  const writable = saved?.writable !== false;
  try {
    const merged = base
      ? mergeDraft(base.source, source, base.state)
      : { state: {}, revised: 0 };
    return {
      state: writable ? merged.state : {},
      revision: useCache ? cached.revision : (saved?.revision ?? 0),
      writable,
      key,
      notice: restoreNotice(saved, merged.revised),
    } satisfies DraftSnapshot;
  } catch {
    return {
      state: {},
      revision: saved?.revision ?? 0,
      writable,
      key,
      notice:
        "The previous draft could not be restored. The current view is shown.",
    } satisfies DraftSnapshot;
  }
}

function restoreNotice(saved: SavedDraft | null, revised: number) {
  if (saved?.writable === false) return SUPERSEDED_NOTICE;
  if (!saved)
    return "Saved draft could not be loaded. Your changes will stay here until sync is available.";
  if (revised)
    return `Updated ${revised} field${revised === 1 ? "" : "s"} from the revised plan; unchanged edits were kept.`;
  return "";
}
