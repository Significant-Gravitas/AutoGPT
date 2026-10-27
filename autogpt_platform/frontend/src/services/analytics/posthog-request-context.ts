import { ImpersonationState } from "@/lib/impersonation";
import { hasConsentFor } from "@/services/consent/consent";
import posthog, { type CaptureResult } from "posthog-js";

export const POSTHOG_SESSION_HEADER_NAME = "X-PostHog-Session-Id";

// PostHog `before_send` hook. While an admin acts as a user, the browser is
// still the admin's (identified as the admin), so its events land on the
// admin's person; the mark lets a project filter leave them out of every
// insight. The backend marks the events it sends during impersonation the
// same way. Nothing here sends an event: capture only runs with analytics
// consent.
export function markImpersonatedEvent(event: CaptureResult | null) {
  if (!event) return event;
  const impersonatedUserId = ImpersonationState.get();
  if (!impersonatedUserId) return event;
  return {
    ...event,
    properties: {
      ...event.properties,
      impersonated: true,
      impersonated_user_id: impersonatedUserId,
    },
  };
}

// The browser's PostHog session as a request header, so the events the
// backend sends for the request join the same session. Only with analytics
// consent, and not while an admin acts as a user: the session is the admin's,
// and the backend's events are the user's.
export function getPostHogSessionHeader(): Record<string, string> {
  if (typeof window === "undefined" || !posthog.__loaded) return {};
  if (!hasConsentFor("analytics") || ImpersonationState.get()) return {};
  try {
    const sessionID = posthog.get_session_id();
    return sessionID ? { [POSTHOG_SESSION_HEADER_NAME]: sessionID } : {};
  } catch {
    return {};
  }
}
