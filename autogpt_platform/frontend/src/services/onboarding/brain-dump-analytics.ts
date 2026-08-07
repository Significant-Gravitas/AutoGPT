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
  // Wizard funnel: every step actually seen (including back-nav and
  // resume-after-refresh), each step left forward, and the whole wizard
  // finished (VISIT_COPILOT posted). The step is in the event NAME
  // ("Onboarding Role Viewed") so the activity feed reads without
  // clicking into properties; the `step` property still rides along for
  // breakdowns. Names are semantic — layouts shuffle indices when the
  // paywall flag flips, names never do.
  | `Onboarding ${string} Viewed`
  | `Onboarding ${string} Completed`
  | "Onboarding Completed"
  | "Brain Dump Started"
  | "Brain Dump Completed"
  | "Brain Dump Canceled"
  | "Brain Dump Skipped"
  | "Brain Dump Recovery Shown"
  | "Brain Dump Recovery Used"
  | "Brain Dump Retry"
  | "Brain Dump Restarted"
  | "Brain Dump Downloaded"
  | "Brain Dump Permission Denied"
  | "Brain Dump Typed Fallback"
  // Wall-clock of the whole finalize round trip — upload flush, virus
  // scan, storage, transcription and extraction. Named for what it
  // actually measures: the client cannot see the transcription step on
  // its own, so calling this "transcription latency" overstated it.
  | "Brain Dump Finalize Latency"
  | "Intro Card Dismissed"
  // The welcome dialog shown on first copilot landing was closed — the
  // greeting fetch and reveal animation start from this moment.
  | "Welcome Dialog Closed"
  // Capability-cards first-run funnel: which cards were reached and how
  // the modal ended (finished the deck vs skipped at card_index).
  | "Capability Card Viewed"
  | "Capability Cards Completed"
  | "Capability Cards Skipped"
  // Connect-tools funnel inside the welcome dialog: CTA opened the
  // picker, a provider row was chosen, a credential actually landed.
  | "Connect Tools Opened"
  | "Connect Tools Provider Selected"
  | "Connect Tools Connected"
  // Which suggested prompt row started the conversation — the generic
  // intro_followup_sent can't tell the rows apart.
  | "Intro Prompt Clicked"
  | "Intro Transcript Copied"
  | "Brain Dump Transcription Failed"
  | "Intro Path Assigned"
  // The user's first real message after seeing the intro card — the
  // signal that the card actually started a conversation. The suggested
  // prompts rendered beneath it are personalised from the same dump, so
  // this covers both a suggestion click and a typed reply.
  | "Intro Followup Sent"
  | "Later Dump Completed";

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

// Super properties ride on every subsequent event this session — how the
// intro path (A/B) becomes filterable on the whole downstream funnel
// instead of living only on the one intro_path event.
export function registerBrainDumpContext(properties: Record<string, unknown>) {
  try {
    posthog.register(properties);
  } catch {
    // Same rule: analytics failures stay invisible.
  }
}
