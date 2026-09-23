import posthog from "posthog-js";

// posthog.init runs in the provider's mount effect, and React runs a child's
// effects before its parent's. So an event captured from a page's own mount
// effect on a full page load (a first visit, or the return from Stripe
// Checkout) reaches posthog-js before init, and posthog-js drops it. Hold such
// events until init has run, then send them with the time they happened.
//
// The queue is bounded, and given up on after a few seconds: with PostHog
// disabled or blocked, init never completes.
const MAX_QUEUED_EVENTS = 50;
const FLUSH_INTERVAL_MS = 250;
const MAX_FLUSH_ATTEMPTS = 40;

interface QueuedEvent {
  event: string;
  properties: Record<string, unknown>;
  timestamp: Date;
}

let queue: QueuedEvent[] = [];
let flushTimer: ReturnType<typeof setInterval> | null = null;
let flushAttempts = 0;

export function capturePostHogEvent(
  event: string,
  properties: Record<string, unknown> = {},
) {
  try {
    if (posthog.__loaded) {
      posthog.capture(event, properties);
      return;
    }
    if (queue.length >= MAX_QUEUED_EVENTS) return;
    queue.push({ event, properties, timestamp: new Date() });
    scheduleFlush();
  } catch {
    // Analytics must never break the screen that reports it.
  }
}

function scheduleFlush() {
  if (flushTimer) return;
  flushAttempts = 0;
  flushTimer = setInterval(flushWhenLoaded, FLUSH_INTERVAL_MS);
}

function flushWhenLoaded() {
  flushAttempts += 1;
  if (posthog.__loaded) {
    const pending = queue;
    queue = [];
    stopFlushing();
    for (const { event, properties, timestamp } of pending) {
      try {
        posthog.capture(event, properties, { timestamp });
      } catch {
        // Keep sending the rest.
      }
    }
    return;
  }
  if (flushAttempts >= MAX_FLUSH_ATTEMPTS) {
    queue = [];
    stopFlushing();
  }
}

function stopFlushing() {
  if (flushTimer) clearInterval(flushTimer);
  flushTimer = null;
}

export function resetPostHogCaptureQueueForTests() {
  queue = [];
  stopFlushing();
}
