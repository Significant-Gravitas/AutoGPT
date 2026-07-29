import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";

// Single source of truth for the payment-lapse markers the backend stamps:
// schedules carry a lowercase free-form `paused_reason`, presets an uppercase
// `PresetDeactivationReason` enum. Keeping them here avoids the string drifting
// across the schedule and trigger components that render the paused state.
export const SCHEDULE_PAYMENT_LAPSED_REASON = "payment_lapsed";
export const PRESET_PAYMENT_LAPSED_REASON = "PAYMENT_LAPSED";

const PAUSED_PAYMENT_REQUIRED_LABEL = "Paused — payment required";
const PAUSED_LABEL = "Paused";

export function getSchedulePausedLabel(pausedReason?: string | null): string {
  return pausedReason === SCHEDULE_PAYMENT_LAPSED_REASON
    ? PAUSED_PAYMENT_REQUIRED_LABEL
    : PAUSED_LABEL;
}

export type WebhookTriggerStatus = "active" | "inactive" | "paused" | "broken";

export function getWebhookTriggerStatus(
  preset: LibraryAgentPreset,
): WebhookTriggerStatus {
  if (!preset.webhook_id || !preset.webhook) return "broken";
  if (preset.is_active) return "active";
  return preset.deactivation_reason === PRESET_PAYMENT_LAPSED_REASON
    ? "paused"
    : "inactive";
}
