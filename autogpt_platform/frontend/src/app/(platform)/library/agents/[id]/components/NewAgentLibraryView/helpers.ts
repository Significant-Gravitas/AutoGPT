import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";

export const AGENT_LIBRARY_SECTION_PADDING_X = "px-4";

export type TriggerKind = "trigger-agent" | "webhook-trigger";

export type SelectedTriggerKind =
  | TriggerKind
  | "loading"
  | "error"
  | "not-found";

// Optional type-hint prefixes for `activeItem` values on the Triggers tab,
// so the right detail view can render without waiting for the item lists.
// Bare IDs (old links, other tabs) parse with a null hint and are resolved
// by list membership instead. IDs are UUIDs, so the prefixes can't collide.
const TRIGGER_KIND_PREFIX: Record<TriggerKind, string> = {
  "trigger-agent": "agent:",
  "webhook-trigger": "preset:",
};

export function parseActiveItemParam(activeItem: string | null): {
  activeItemId: string | null;
  triggerKindHint: TriggerKind | null;
} {
  const entries = Object.entries(TRIGGER_KIND_PREFIX) as [
    TriggerKind,
    string,
  ][];
  for (const [kind, prefix] of entries) {
    if (activeItem?.startsWith(prefix)) {
      return {
        activeItemId: activeItem.slice(prefix.length),
        triggerKindHint: kind,
      };
    }
  }
  return { activeItemId: activeItem, triggerKindHint: null };
}

export function activeItemParamFor(kind: TriggerKind, id: string): string {
  return `${TRIGGER_KIND_PREFIX[kind]}${id}`;
}

/** Presets with a webhook show under "Triggers"; the rest are templates. */
export function isWebhookPreset(preset: LibraryAgentPreset): boolean {
  return !!preset.webhook_id;
}

/**
 * Resolve what a selected Triggers-tab item actually is. An unknown ID must
 * never be assumed to be a preset: fetching a preset by a trigger-agent ID
 * (or a stale link) guarantees a 404 error screen. List membership is the
 * source of truth; the URL hint only short-circuits the loading state, so a
 * wrong or stale hint self-corrects once the lists load.
 */
export function deriveSelectedTriggerKind(args: {
  activeItemId: string | null;
  triggerKindHint: TriggerKind | null;
  triggerAgents: Pick<LibraryAgent, "id">[] | undefined;
  presets: LibraryAgentPreset[] | undefined;
  presetsComplete: boolean;
  listsResolved: boolean;
  anyListFailed: boolean;
}): SelectedTriggerKind | null {
  if (!args.activeItemId) return null;
  if (args.triggerAgents?.some((t) => t.id === args.activeItemId)) {
    return "trigger-agent";
  }
  if (
    args.presets?.some((p) => isWebhookPreset(p) && p.id === args.activeItemId)
  ) {
    return "webhook-trigger";
  }
  // Membership is only conclusive once both lists resolved successfully.
  if (!args.listsResolved) {
    if (args.triggerKindHint) return args.triggerKindHint;
    return args.anyListFailed ? "error" : "loading";
  }
  // If the fetched presets page is capped/incomplete, the ID could be a
  // preset beyond it — let the by-ID detail view resolve it (fails fast
  // into the not-found card if truly gone).
  if (!args.presetsComplete) return "webhook-trigger";
  return "not-found";
}

export function getErrorStatus(error: unknown): number | null {
  if (typeof error !== "object" || error === null || !("status" in error)) {
    return null;
  }
  const status = (error as { status?: unknown }).status;
  return typeof status === "number" ? status : null;
}

/** A 4xx won't heal on retry — fail fast instead of stalling ~7s in backoff. */
export function retryUnlessClientError(
  failureCount: number,
  error: unknown,
): boolean {
  const status = getErrorStatus(error);
  return (
    failureCount < 3 && !(status !== null && status >= 400 && status < 500)
  );
}
