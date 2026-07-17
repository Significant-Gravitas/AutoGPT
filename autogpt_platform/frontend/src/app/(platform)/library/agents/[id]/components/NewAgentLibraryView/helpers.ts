export const AGENT_LIBRARY_SECTION_PADDING_X = "px-4";

export type TriggerKind = "trigger-agent" | "webhook-trigger";

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
  if (activeItem?.startsWith("agent:")) {
    return {
      activeItemId: activeItem.slice("agent:".length),
      triggerKindHint: "trigger-agent",
    };
  }
  if (activeItem?.startsWith("preset:")) {
    return {
      activeItemId: activeItem.slice("preset:".length),
      triggerKindHint: "webhook-trigger",
    };
  }
  return { activeItemId: activeItem, triggerKindHint: null };
}
