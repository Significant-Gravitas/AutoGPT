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
 */
export function parseActiveItemParam(activeItem: string | null): {
  activeItemId: string | null;
  triggerKindHint: TriggerKind | null;
} {
  for (const [kind, prefix] of Object.entries(TRIGGER_KIND_PREFIX)) {
    if (activeItem?.startsWith(prefix)) {
      return {
        activeItemId: activeItem.slice(prefix.length),
        triggerKindHint: kind as TriggerKind,
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

export function isClientError(error: unknown): boolean {
  if (typeof error !== "object" || error === null || !("status" in error)) {
    return false;
  }
  const status = (error as { status?: unknown }).status;
  return typeof status === "number" && status >= 400 && status < 500;
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
