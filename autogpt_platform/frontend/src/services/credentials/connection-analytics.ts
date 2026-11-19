// Terminal states of a credential connect card. Every one of these ends in a
// missing row, a disabled button or a spinner rather than an exception, so
// nothing reports them today — the backend's Sentry events stop at the point
// the credential is stored, and the failures below all happen after that.
//
// `failure_class` matches the tag the backend puts on its own credential
// failures, so one class number covers both halves of a path.

import {
  CredentialConnectionFailureEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

type CredentialConnectionFailure = EventName<
  typeof CredentialConnectionFailureEvent
>;

const FAILURE_CLASS: Record<CredentialConnectionFailure, string> = {
  [CredentialConnectionFailureEvent.CREDENTIAL_CARD_NEVER_RENDERED]:
    "class_03_provider_unknown_to_frontend",
  [CredentialConnectionFailureEvent.CREDENTIAL_OAUTH_POPUP_BLOCKED]:
    "class_05_browser_channel_broken",
  [CredentialConnectionFailureEvent.CREDENTIAL_OAUTH_FLOW_TIMED_OUT]:
    "class_05_browser_channel_broken",
  [CredentialConnectionFailureEvent.CREDENTIAL_SCOPE_SHORTFALL_BLOCKED_SELECTION]:
    "class_08_scopes_too_narrow",
  [CredentialConnectionFailureEvent.CREDENTIAL_PROCEED_STUCK_AFTER_CONNECT]:
    "class_11_credential_not_wired_to_card",
};

export function trackCredentialConnectionFailure(
  event: CredentialConnectionFailure,
  properties: { provider?: string } & Record<string, unknown> = {},
) {
  try {
    posthog.capture(event, {
      ...properties,
      failure_class: FAILURE_CLASS[event],
    });
  } catch {
    // A blocked analytics host must never break a connect flow.
  }
}
