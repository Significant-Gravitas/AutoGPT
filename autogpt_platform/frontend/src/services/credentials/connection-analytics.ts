// Terminal states of a credential connect card. Every one of these ends in a
// missing row, a disabled button or a spinner rather than an exception, so
// nothing reports them today — the backend's Sentry events stop at the point
// the credential is stored, and the failures below all happen after that.
//
// `failure_class` matches the tag the backend puts on its own credential
// failures, so one class number covers both halves of a path.

import posthog from "posthog-js";

type CredentialConnectionFailure =
  // The provider is absent from the loaded provider map, so the card's row
  // for it never renders at all.
  | "credential_card_never_rendered"
  // Popup and the new-tab fallback were both blocked: there is no way in.
  | "credential_oauth_popup_blocked"
  | "credential_oauth_flow_timed_out"
  // Connected, stored, and then refused by the card because the provider
  // granted less than the block asked for.
  | "credential_scope_shortfall_blocked_selection"
  // A sign-in completed on this card and the card is still not ready.
  | "credential_proceed_stuck_after_connect"
  // Proceed offered on a chain restored from history, where it drafts a
  // "I've configured the required credentials" reply about nothing.
  | "credential_proceed_stale_from_history";

const FAILURE_CLASS: Record<CredentialConnectionFailure, string> = {
  credential_card_never_rendered: "class_03_provider_unknown_to_frontend",
  credential_oauth_popup_blocked: "class_05_browser_channel_broken",
  credential_oauth_flow_timed_out: "class_05_browser_channel_broken",
  credential_scope_shortfall_blocked_selection: "class_08_scopes_too_narrow",
  credential_proceed_stuck_after_connect:
    "class_11_credential_not_wired_to_card",
  credential_proceed_stale_from_history: "class_13_chain_turn_mismatch",
};

export function trackCredentialConnectionFailure(
  event: CredentialConnectionFailure,
  properties: { provider?: string } & Record<string, unknown> = {},
) {
  try {
    posthog.capture(event, {
      failure_class: FAILURE_CLASS[event],
      ...properties,
    });
  } catch {
    // A blocked analytics host must never break a connect flow.
  }
}
