// Brain-dump funnel events. These are the release-blocker instrumentation:
// completion rate, dump length distribution, and every way a dump can go
// wrong. Capture is best-effort — a blocked analytics host must never
// interrupt someone mid-recording.
//
// Lives in services/ rather than beside the onboarding step because the
// copilot home fires the tail of this funnel (the intro and the follow-up
// dump), and a feature folder reaching into another feature's internals
// is how import graphs rot.

import posthog from "posthog-js";

type BrainDumpEvent =
  | "brain_dump_started"
  | "brain_dump_completed"
  | "brain_dump_canceled"
  | "brain_dump_skipped"
  | "brain_dump_recovery_shown"
  | "brain_dump_recovery_used"
  | "brain_dump_retry"
  | "brain_dump_restarted"
  | "brain_dump_download"
  | "brain_dump_permission_denied"
  | "brain_dump_typed_fallback"
  // Wall-clock of the whole finalize round trip — upload flush, virus
  // scan, storage, transcription and extraction. Named for what it
  // actually measures: the client cannot see the transcription step on
  // its own, so calling this "transcription latency" overstated it.
  | "finalize_latency_ms"
  | "intro_card_dismissed"
  // The welcome dialog shown on first copilot landing was closed — the
  // greeting fetch and reveal animation start from this moment.
  | "welcome_dialog_closed"
  // Capability-cards first-run funnel: which cards were reached and how
  // the modal ended (finished the deck vs skipped at card_index).
  | "capability_card_viewed"
  | "capability_cards_completed"
  | "capability_cards_skipped"
  | "transcription_failed"
  | "intro_path"
  // The user's first real message after seeing the intro card — the
  // signal that the card actually started a conversation. The suggested
  // prompts rendered beneath it are personalised from the same dump, so
  // this covers both a suggestion click and a typed reply.
  | "intro_followup_sent"
  | "later_dump_completed"
  // The team Otto proposed on the greeting page: one event per card
  // shown, then the doors out of it — hire, raise your own, talk it
  // through, or skip straight to the builder.
  | "expert_recommended"
  | "expert_recommendation_clicked"
  | "hire_started"
  | "raise_door_clicked"
  | "intro_start_with_autopilot"
  // The wizard's hire step: a hire that landed from a card, and the step
  // being left — with how many of the proposed experts were hired, so the
  // funnel can tell "hired a team" from "skipped past it".
  | "onboarding_expert_hired"
  | "hire_step_continued";

export function trackBrainDump(
  event: BrainDumpEvent,
  properties?: Record<string, unknown>,
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // Analytics is never worth a broken recording.
  }
}
