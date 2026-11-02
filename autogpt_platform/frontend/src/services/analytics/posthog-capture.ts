import {
  getConsentAnswer,
  isAwaitingConsentAnswer,
  isConsentManagerConfigured,
  subscribeToConsent,
  subscribeToConsentPrompt,
} from "@/services/consent/consent";
import posthog from "posthog-js";

// PostHog starts opted out and is opted in once the visitor's analytics
// consent is known (providers/posthog/posthog-consent.ts). An event captured
// before that, e.g. from a page's mount effect on a full page load, would be
// dropped by posthog-js, so it is held here until the consent is resolved:
//
// - analytics granted (an earlier answer, or a region Cookiebot auto-consents
//   once uc.js loads): sent with the time it happened once PostHog captures;
// - denied, no consent manager, or the banner asking: dropped. An answer
//   given on the banner covers what happens after it, never what was held
//   before it.
//
// The queue is bounded, and dropped after a few seconds if nothing resolves
// (uc.js blocked, or PostHog disabled or blocked).
const MAX_QUEUED_EVENTS = 50;
const FLUSH_INTERVAL_MS = 250;
const MAX_FLUSH_ATTEMPTS = 40;
// Between uc.js loading and it deciding whether this visitor needs a banner,
// the API is there without an answer. Past this grace it is the banner, even
// if its dialog events never reached us.
const PROMPT_GRACE_ATTEMPTS = 4;

interface CaptureOptions {
  /**
   * Send at most once per tab. Marked when the event is sent or deliberately
   * dropped, so an event lost while held is tried again on the next mount.
   */
  oncePerTabKey?: string;
}

interface QueuedEvent {
  event: string;
  properties: Record<string, unknown>;
  timestamp: Date;
  oncePerTabKey?: string;
}

type Decision = "send" | "hold" | "drop";

let queue: QueuedEvent[] = [];
let flushTimer: ReturnType<typeof setInterval> | null = null;
let flushAttempts = 0;
let awaitingAnswerAttempts = 0;
let promptShown = false;
let stopFollowingConsent: (() => void) | null = null;

export function capturePostHogEvent(
  event: string,
  properties: Record<string, unknown> = {},
  { oncePerTabKey }: CaptureOptions = {},
) {
  try {
    if (oncePerTabKey && isAlreadyHandled(oncePerTabKey)) return;
    const decision = decideNow();
    if (decision === "send") {
      posthog.capture(event, properties);
      markHandled(oncePerTabKey);
      return;
    }
    if (decision === "drop") {
      markHandled(oncePerTabKey);
      return;
    }
    if (queue.length >= MAX_QUEUED_EVENTS) return;
    queue.push({ event, properties, timestamp: new Date(), oncePerTabKey });
    startHolding();
  } catch {
    // Analytics must never break the screen that reports it.
  }
}

function isCapturing() {
  return Boolean(posthog.__loaded && posthog.is_capturing());
}

// For an event captured now. The banner already asking means the visitor
// has not answered; anything after this capture is their fresh decision.
function decideNow(): Decision {
  if (isCapturing()) return "send";
  const answer = getConsentAnswer();
  if (answer) return answer.analytics ? "hold" : "drop";
  if (!isConsentManagerConfigured() || isAwaitingConsentAnswer()) return "drop";
  return "hold";
}

// For the held queue: Cookiebot was not loaded when it started, so an API
// without an answer may still be deciding whether this visitor needs asking.
function decideHeld(): Decision {
  if (isCapturing()) return "send";
  const answer = getConsentAnswer();
  if (answer) return answer.analytics ? "hold" : "drop";
  if (promptShown || !isConsentManagerConfigured()) return "drop";
  if (isAwaitingConsentAnswer()) {
    awaitingAnswerAttempts += 1;
    return awaitingAnswerAttempts > PROMPT_GRACE_ATTEMPTS ? "drop" : "hold";
  }
  // uc.js not loaded yet. A blocked one never answers; the timeout drops.
  return "hold";
}

function startHolding() {
  if (flushTimer) return;
  flushAttempts = 0;
  awaitingAnswerAttempts = 0;
  promptShown = false;
  flushTimer = setInterval(onFlushTick, FLUSH_INTERVAL_MS);
  const stopConsent = subscribeToConsent(resolveHeld);
  const stopPrompt = subscribeToConsentPrompt(onPrompt);
  stopFollowingConsent = () => {
    stopConsent();
    stopPrompt();
  };
}

function onPrompt() {
  promptShown = true;
  resolveHeld();
}

function onFlushTick() {
  flushAttempts += 1;
  resolveHeld();
  if (queue.length && flushAttempts >= MAX_FLUSH_ATTEMPTS) finish("drop");
}

function resolveHeld() {
  try {
    const decision = decideHeld();
    if (decision !== "hold") finish(decision);
  } catch {
    finish("drop");
  }
}

function finish(decision: Exclude<Decision, "hold">) {
  const pending = queue;
  queue = [];
  stopHolding();
  for (const { event, properties, timestamp, oncePerTabKey } of pending) {
    if (decision === "send") {
      try {
        posthog.capture(event, properties, { timestamp });
      } catch {
        // Keep sending the rest.
      }
    }
    markHandled(oncePerTabKey);
  }
}

function stopHolding() {
  if (flushTimer) clearInterval(flushTimer);
  flushTimer = null;
  stopFollowingConsent?.();
  stopFollowingConsent = null;
}

function isAlreadyHandled(key: string) {
  if (queue.some((queued) => queued.oncePerTabKey === key)) return true;
  try {
    return Boolean(sessionStorage.getItem(key));
  } catch {
    // In-app browsers may block sessionStorage; double-counting beats dropping.
    return false;
  }
}

function markHandled(key: string | undefined) {
  if (!key) return;
  try {
    sessionStorage.setItem(key, "1");
  } catch {
    // See isAlreadyHandled.
  }
}

export function resetPostHogCaptureQueueForTests() {
  queue = [];
  stopHolding();
}
