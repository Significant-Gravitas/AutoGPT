import { hasConsentFor, subscribeToConsent } from "@/services/consent/consent";
import * as Sentry from "@sentry/nextjs";

function createReplayIntegrations() {
  // Canvas first: Replay looks it up by name when it sets up.
  return [
    Sentry.replayCanvasIntegration(),
    Sentry.replayIntegration({
      unmask: [".sentry-unmask, [data-sentry-unmask]"],
    }),
  ];
}

/**
 * Session replay records the screen, so it only exists once the visitor has
 * consented to monitoring. With consent already stored the integrations go
 * into Sentry.init; otherwise they are added the moment consent is granted,
 * without a reload, and nothing is recorded before that.
 */
export function setupSessionReplay() {
  if (hasConsentFor("monitoring")) return createReplayIntegrations();

  const unsubscribe = subscribeToConsent(() => {
    if (!hasConsentFor("monitoring")) return;
    unsubscribe();
    createReplayIntegrations().forEach((integration) =>
      Sentry.addIntegration(integration),
    );
  });
  return [];
}
