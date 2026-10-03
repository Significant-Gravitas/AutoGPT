// Every PostHog event name the browser sends, in one place. What each one
// means, who sends it and which properties it carries is in
// docs/platform/tracking-plan.md; the backend's list is
// backend/util/posthog_events.py.
//
// The names follow the product analytics plan ("Every Second Counts"). Never
// change a value here. PostHog stores the raw string, so a rename orphans
// every insight and funnel built on the old name and cannot be backfilled.
// __tests__/posthog-events.test.ts pins the values and reserves the names
// retired or renamed in SECRT-2722.
//
// Plain `as const` objects rather than enums so a string literal at a call
// site still type-checks against the list. Call sites do pass literals, so
// a rename has to search for the old string beyond this file.

export const PageEvent = {
  PAGEVIEW: "$pageview",
} as const;

// Ops signal from the dual flag backend, not a user action: LaunchDarkly and
// PostHog resolved the same flag to different values.
export const FeatureFlagEvent = {
  FEATURE_FLAG_MISMATCHED: "feature_flag_mismatched",
} as const;

export const ExpertsFunnelEvent = {
  EXPERTS_SECTION_VIEWED: "experts_section_viewed",
  HOME_VIEWED: "home_viewed",
  BRIEFING_OPENED: "briefing_opened",
  EXPERT_PROFILE_OPENED: "expert_profile_opened",
  HIRE_STARTED: "hire_started",
  EXPERT_THREAD_CREATED: "expert_thread_created",
  BRIEFING_OUTCOME_CLICKED: "briefing_outcome_clicked",
  HOME_ATTENTION_ACTIONED: "home_attention_actioned",
  HOME_TEAM_MEMBER_CLICKED: "home_team_member_clicked",
} as const;

export const HireFlowEvent = {
  HIRE_FLOW_ABANDONED: "hire_flow_abandoned",
} as const;

export const BrainDumpEvent = {
  BRAIN_DUMP_STARTED: "brain_dump_started",
  // `finalize_latency_ms` on this and on `transcription_failed` is the
  // wall-clock of the `finalizeBrainDump()` round trip: virus scan, storage,
  // transcription and extraction. The upload flush finishes before it starts.
  BRAIN_DUMP_COMPLETED: "brain_dump_completed",
  BRAIN_DUMP_CANCELED: "brain_dump_canceled",
  BRAIN_DUMP_SKIPPED: "brain_dump_skipped",
  BRAIN_DUMP_RECOVERY_SHOWN: "brain_dump_recovery_shown",
  BRAIN_DUMP_RECOVERY_USED: "brain_dump_recovery_used",
  BRAIN_DUMP_RETRY: "brain_dump_retry",
  BRAIN_DUMP_RESTARTED: "brain_dump_restarted",
  BRAIN_DUMP_DOWNLOAD: "brain_dump_download",
  BRAIN_DUMP_PERMISSION_DENIED: "brain_dump_permission_denied",
  BRAIN_DUMP_TYPED_FALLBACK: "brain_dump_typed_fallback",
  // The welcome dialog shown on first copilot landing was closed — the
  // greeting fetch and reveal animation start from this moment.
  WELCOME_DIALOG_CLOSED: "welcome_dialog_closed",
  // Capability-cards first-run funnel: which cards were reached and how
  // the modal ended (finished the deck vs skipped at card_index).
  CAPABILITY_CARD_VIEWED: "capability_card_viewed",
  CAPABILITY_CARDS_COMPLETED: "capability_cards_completed",
  CAPABILITY_CARDS_SKIPPED: "capability_cards_skipped",
  TRANSCRIPTION_FAILED: "transcription_failed",
  INTRO_PATH: "intro_path",
  // The user's first real message after seeing the intro card — the
  // signal that the card actually started a conversation. The suggested
  // prompts rendered beneath it are personalised from the same dump, so
  // this covers both a suggestion click and a typed reply.
  INTRO_FOLLOWUP_SENT: "intro_followup_sent",
  LATER_DUMP_COMPLETED: "later_dump_completed",
  // The team Otto proposed on the greeting page: one event per card
  // shown, then the doors out of it — hire, create your own, talk it
  // through, or skip straight to the builder.
  EXPERT_RECOMMENDED: "expert_recommended",
  EXPERT_RECOMMENDATION_CLICKED: "expert_recommendation_clicked",
  // The wizard's hire step was left, with how many of the proposed experts
  // were hired, so the funnel can tell "hired a team" from "skipped past it".
  // The hires themselves are the backend's `expert_hired`.
  HIRE_STEP_CONTINUED: "hire_step_continued",
} as const;

export const TabIntroEvent = {
  TAB_INTRO_SHOWN: "tab_intro_shown",
  // The card's primary CTA was used, as opposed to any of the ways out
  // ("Got it", Escape, the backdrop) that all land on `tab_intro_dismissed`.
  TAB_INTRO_CTA_CLICKED: "tab_intro_cta_clicked",
  TAB_INTRO_DISMISSED: "tab_intro_dismissed",
} as const;

export const VoiceModeEvent = {
  // Enabled from the composer. `entry` says whether a chat already existed,
  // since starting one costs a session-creation round trip first.
  VOICE_MODE_STARTED: "voice_mode_started",
  // Left deliberately: the toggle, or Stop during a reply.
  VOICE_MODE_STOPPED: "voice_mode_stopped",
  // Closed by the silence timeout rather than by the user. A high share
  // here means the timeout is too short, which is exactly the complaint
  // that moved the VAD from 700 ms to 1540 ms.
  VOICE_MODE_TIMED_OUT: "voice_mode_timed_out",
  // A completed turn: heard, transcribed, sent. `turn_index` counts within
  // the session, so a histogram shows whether anyone gets past one.
  // `transcribe_latency_ms` is speech end to transcript in hand: the number
  // that decides whether streaming STT is worth building.
  VOICE_TURN_SENT: "voice_turn_sent",
  // Heard something and threw it away — VAD misfire, or filler/hallucination
  // like Whisper's "Thank you." on silence. Splits by `reason`.
  VOICE_TURN_DROPPED: "voice_turn_dropped",
  // The mic reopened after a reply finished playing: a full loop closed.
  // `first_sound_latency_ms` is speech end to the first spoken word.
  VOICE_TURN_COMPLETED: "voice_turn_completed",
  // The user asked for a failed transcription to be tried again. Against
  // `voice_turn_dropped{reason:transcribe_failed}` this says how many of those
  // dropped turns the user actually got back.
  VOICE_TRANSCRIBE_RETRIED: "voice_transcribe_retried",
  // Gave up on transcription and took the audio instead. Rare by design: a
  // run of these means retrying is not working.
  VOICE_RECORDING_DOWNLOADED: "voice_recording_downloaded",
  VOICE_MODE_PERMISSION_DENIED: "voice_mode_permission_denied",
  // Synthesis or the VAD failed. `stage` says which.
  VOICE_MODE_ERROR: "voice_mode_error",
} as const;

export const CredentialConnectionFailureEvent = {
  // The provider is absent from the loaded provider map, so the card's row
  // for it never renders at all.
  CREDENTIAL_CARD_NEVER_RENDERED: "credential_card_never_rendered",
  // Popup and the new-tab fallback were both blocked: there is no way in.
  CREDENTIAL_OAUTH_POPUP_BLOCKED: "credential_oauth_popup_blocked",
  CREDENTIAL_OAUTH_FLOW_TIMED_OUT: "credential_oauth_flow_timed_out",
  // Connected, stored, and then refused by the card because the provider
  // granted less than the block asked for.
  CREDENTIAL_SCOPE_SHORTFALL_BLOCKED_SELECTION:
    "credential_scope_shortfall_blocked_selection",
  // A sign-in completed on this card and the credential never reached it.
  CREDENTIAL_PROCEED_STUCK_AFTER_CONNECT:
    "credential_proceed_stuck_after_connect",
} as const;

export const TrialEvent = {
  TRIAL_OFFER_VIEWED: "trial_offer_viewed",
  SUBSCRIPTION_TRIAL_CHECKOUT_STARTED: "subscription_trial_checkout_started",
} as const;

export const PostHogEvent = {
  ...PageEvent,
  ...FeatureFlagEvent,
  ...ExpertsFunnelEvent,
  ...HireFlowEvent,
  ...BrainDumpEvent,
  ...TabIntroEvent,
  ...VoiceModeEvent,
  ...CredentialConnectionFailureEvent,
  ...TrialEvent,
} as const;

// In the tracking plan and NOT sent yet (SECRT-2723). Move a name into its
// group above in the change that starts sending it.
export const PlannedPostHogEvent = {
  ONBOARDING_STEP_VIEWED: "onboarding_step_viewed",
  PAYWALL_VIEWED: "paywall_viewed",
  PLAN_SELECTED: "plan_selected",
  BILLING_PORTAL_OPENED: "billing_portal_opened",
  CHECKOUT_ABANDONED: "checkout_abandoned",
  TOUR_STARTED: "tour_started",
  TOUR_SCENARIO_STARTED: "tour_scenario_started",
  TOUR_SCENARIO_COMPLETED: "tour_scenario_completed",
  TOUR_CTA_CLICKED: "tour_cta_clicked",
} as const;

export type EventName<Group extends Record<string, string>> =
  Group[keyof Group];
