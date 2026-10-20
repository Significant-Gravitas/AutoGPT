import {
  holdsCredential,
  redactEventURLs,
  redactRecordingEventURLs,
} from "@/lib/sentry-url-redaction";
import { hasConsentFor, subscribeToConsent } from "@/services/consent/consent";
import * as Sentry from "@sentry/nextjs";

// How the replay was recording when it was held back for a page whose address
// carries a credential, or "start" when it had not started yet.
let heldBack: "start" | "session" | "buffer" | null = null;
// The stop that held it back, which a restart waits for: it clears the
// recording's state when it settles.
let stopping: Promise<void> = Promise.resolve();

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

/**
 * Stops the replay before a page whose address carries a credential renders.
 * Next calls it as each client navigation starts.
 */
export function holdReplayBackFor(href: string) {
  if (heldBack || !holdsCredential(href)) return;
  const replay = Sentry.getReplay();
  const mode = replay?.getRecordingMode();
  if (!replay?.getReplayId() || !mode) return;
  heldBack = mode;
  stopping = replay.stop().catch(() => undefined);
}

/**
 * Called once a navigation has landed: holds the replay back on a page whose
 * address carries a credential, and otherwise resumes it as it was.
 */
export function syncReplayWithPage(href: string) {
  if (holdsCredential(href)) {
    holdReplayBackFor(href);
    return;
  }
  if (heldBack) void stopping.then(resumeReplay);
}

// Runs once the stop has settled, against the address and consent as they
// are by then.
function resumeReplay() {
  const mode = heldBack;
  if (!mode || holdsCredential(window.location.href)) return;
  heldBack = null;
  if (!hasConsentFor("monitoring")) return;
  if (mode === "start") Sentry.addIntegration(createReplayIntegration());
  else if (mode === "session") Sentry.getReplay()?.start();
  else Sentry.getReplay()?.startBuffering();
}

function createReplayIntegrations() {
  if (!holdsCredential(window.location.href)) {
    return [createReplayIntegration(), replayEventURLRedaction()];
  }
  heldBack = "start";
  return [replayEventURLRedaction()];
}

function createReplayIntegration() {
  // Everything on screen is masked unless an element opts in with
  // `sentry-unmask`, which only copy that is the same for every user should.
  // No canvas recording: its pixels are sent as drawn, with no masking.
  return Sentry.replayIntegration({
    maskAllText: true,
    maskAllInputs: true,
    blockAllMedia: true,
    unmask: [".sentry-unmask, [data-sentry-unmask]"],
    beforeAddRecordingEvent: redactRecordingEventURLs,
  });
}

// Replay events skip beforeSend, so their URLs are masked here instead.
function replayEventURLRedaction() {
  return {
    name: "ReplayEventURLRedaction",
    processEvent(event: Sentry.Event) {
      return event.type === "replay_event" ? redactEventURLs(event) : event;
    },
  };
}
