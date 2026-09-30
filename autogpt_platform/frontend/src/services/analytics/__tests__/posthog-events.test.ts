import { describe, expect, it } from "vitest";
import { PlannedPostHogEvent, PostHogEvent } from "../posthog-events";

// A failure here means an event name changed. PostHog cannot backfill a
// rename, so every insight built on the old name silently goes empty. Add
// new names freely; never edit an existing string.

const LIVE_EVENT_NAMES = [
  "$pageview",
  "brain_dump_canceled",
  "brain_dump_completed",
  "brain_dump_download",
  "brain_dump_permission_denied",
  "brain_dump_recovery_shown",
  "brain_dump_recovery_used",
  "brain_dump_restarted",
  "brain_dump_retry",
  "brain_dump_skipped",
  "brain_dump_started",
  "brain_dump_typed_fallback",
  "briefing_opened",
  "briefing_outcome_clicked",
  "capability_card_viewed",
  "capability_cards_completed",
  "capability_cards_skipped",
  "credential_card_never_rendered",
  "credential_oauth_flow_timed_out",
  "credential_oauth_popup_blocked",
  "credential_proceed_stuck_after_connect",
  "credential_scope_shortfall_blocked_selection",
  "experiment_exposed",
  "expert_profile_opened",
  "expert_recommendation_clicked",
  "expert_recommended",
  "expert_thread_created",
  "experts_section_viewed",
  "feature_flag_mismatch",
  "finalize_latency_ms",
  "hire_flow_abandoned",
  "hire_flow_completed",
  "hire_started",
  "hire_step_continued",
  "home_attention_actioned",
  "home_team_member_clicked",
  "home_viewed",
  "intro_card_dismissed",
  "intro_followup_sent",
  "intro_path",
  "intro_start_with_autopilot",
  "later_dump_completed",
  "onboarding_expert_hired",
  "raise_door_clicked",
  "subscription_trial_checkout_started",
  "subscription_trial_offer_viewed",
  "tab_intro_cta_clicked",
  "tab_intro_dismissed",
  "tab_intro_shown",
  "transcription_failed",
  "voice_first_sound_latency_ms",
  "voice_mode_error",
  "voice_mode_permission_denied",
  "voice_mode_started",
  "voice_mode_stopped",
  "voice_mode_timed_out",
  "voice_recording_downloaded",
  "voice_transcribe_latency_ms",
  "voice_transcribe_retried",
  "voice_turn_completed",
  "voice_turn_dropped",
  "voice_turn_sent",
  "welcome_dialog_closed",
];

const PLANNED_EVENT_NAMES = [
  "billing_portal_opened",
  "checkout_abandoned",
  "onboarding_step_viewed",
  "paywall_viewed",
  "plan_selected",
  "tour_cta_clicked",
  "tour_scenario_completed",
  "tour_scenario_started",
  "tour_started",
];

describe("PostHog event names", () => {
  it("keeps every live name exactly as PostHog already stores it", () => {
    expect(Object.values(PostHogEvent).sort()).toEqual(LIVE_EVENT_NAMES);
  });

  it("keeps the planned names that are not sent yet", () => {
    expect(Object.values(PlannedPostHogEvent).sort()).toEqual(
      PLANNED_EVENT_NAMES,
    );
  });

  it("never lists a planned name as live", () => {
    const live = new Set<string>(Object.values(PostHogEvent));
    expect(PLANNED_EVENT_NAMES.filter((name) => live.has(name))).toEqual([]);
  });
});
