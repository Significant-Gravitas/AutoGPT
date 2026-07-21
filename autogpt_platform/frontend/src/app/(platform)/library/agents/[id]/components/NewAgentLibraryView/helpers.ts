import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";

export const AGENT_LIBRARY_SECTION_PADDING_X = "px-4";

export type TriggerKind = "trigger-agent" | "webhook-trigger";

export type SelectedTriggerKind =
  | TriggerKind
  | "loading"
  | "error"
  | "not-found";

const TRIGGER_KIND_PREFIX: Record<TriggerKind, string> = {
  "trigger-agent": "agent:",
  "webhook-trigger": "preset:",
};

/**
 * The `activeItem` URL param may carry a type hint for items on the Triggers
 * tab — `agent:<id>` for trigger agents, `preset:<id>` for webhook-trigger
 * presets — so the right detail view can render without waiting for the
 * trigger-agent and preset lists to load. Bare IDs (old links, other tabs)
 * parse with a null hint and are resolved by list membership instead.
 *
 * The prefix tokens can't collide with real IDs: item IDs are UUIDs, which
 * never contain `:` nor start with `agent:`/`preset:`.
 */
export function parseActiveItemParam(activeItem: string | null): {
  activeItemId: string | null;
  triggerKindHint: TriggerKind | null;
} {
  const prefixEntries = Object.entries(TRIGGER_KIND_PREFIX) as [
    TriggerKind,
    string,
  ][];
  for (const [kind, prefix] of prefixEntries) {
    if (activeItem?.startsWith(prefix)) {
      return {
        activeItemId: activeItem.slice(prefix.length),
        triggerKindHint: kind,
      };
    }
  }
  return { activeItemId: activeItem, triggerKindHint: null };
}

/** Inverse of {@link parseActiveItemParam}: build a type-hinted `activeItem` value. */
export function activeItemParamFor(kind: TriggerKind, id: string): string {
  return `${TRIGGER_KIND_PREFIX[kind]}${id}`;
}

/** Presets with a webhook show under "Triggers"; the rest are templates. */
export function isWebhookPreset(preset: LibraryAgentPreset): boolean {
  return !!preset.webhook_id;
}

/**
 * Resolve what a selected Triggers-tab item actually is. An unknown ID must
 * never be assumed to be a preset: firing a preset fetch for a trigger-agent
 * ID (or a stale link) guarantees a 404 error screen. List membership is the
 * source of truth; the URL's `agent:`/`preset:` hint only short-circuits the
 * loading state, so a wrong or stale hint still resolves to the right view
 * (or "not-found") once the lists load.
 */
export function deriveSelectedTriggerKind(args: {
  activeItemId: string | null;
  triggerKindHint: TriggerKind | null;
  triggerAgents: Pick<LibraryAgent, "id">[] | undefined;
  presets: LibraryAgentPreset[] | undefined;
  /** Whether the fetched presets page is the complete set for this agent. */
  presetsComplete: boolean;
  /** Whether both list queries have settled successfully. */
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

  // Membership is only conclusive once both lists have resolved: a pending
  // or failed fetch says nothing about whether the item exists.
  if (!args.listsResolved) {
    // Deliberate optimistic routing: trusting the hint here means a wrong
    // or stale hint costs one throwaway, fast-failing by-ID fetch before
    // membership self-corrects — the price of instant deep-link rendering.
    if (args.triggerKindHint) return args.triggerKindHint;
    return args.anyListFailed ? "error" : "loading";
  }
  // The fetched presets page may be capped; if it isn't the complete set,
  // the ID could be a preset beyond the first page — let the preset detail
  // view resolve it by ID (it fails fast into the not-found card if gone).
  if (!args.presetsComplete) return "webhook-trigger";
  return "not-found";
}

function getErrorStatus(error: unknown): number | null {
  if (typeof error !== "object" || error === null || !("status" in error)) {
    return null;
  }
  const status = (error as { status?: unknown }).status;
  return typeof status === "number" ? status : null;
}

export function isClientError(error: unknown): boolean {
  const status = getErrorStatus(error);
  return status !== null && status >= 400 && status < 500;
}

export function isNotFoundError(error: unknown): boolean {
  return getErrorStatus(error) === 404;
}

/**
 * Retry policy for fetch-by-ID queries: a 4xx (e.g. 404 for a deleted item)
 * won't heal on retry, so fail fast instead of stalling ~7s in backoff.
 */
export function retryUnlessClientError(
  failureCount: number,
  error: unknown,
): boolean {
  return failureCount < 3 && !isClientError(error);
}
