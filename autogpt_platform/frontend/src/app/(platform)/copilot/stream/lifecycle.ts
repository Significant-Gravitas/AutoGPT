/**
 * The one reconnect state machine of a chat's stream. Pure: the hook that
 * runs it (`useCopilotStreamLifecycle`) feeds it inputs and carries out the
 * effect each transition returns.
 *
 * Only a stream that stopped carrying frames is a dead one. The server writes
 * a heartbeat every `HEARTBEAT_INTERVAL_MS`, so a model or tool that is silent
 * for minutes is still a live stream; "still working" is the StreamStatus UX,
 * never a reconnect.
 */
export const HEARTBEAT_INTERVAL_MS = 10_000;
export const HEARTBEAT_MISS_MS = 3 * HEARTBEAT_INTERVAL_MS;
/** A tab hidden at least this long checks for a turn that started meanwhile. */
export const WAKE_RESYNC_THRESHOLD_MS = 30_000;
const BACKOFF_BASE_MS = 500;
const BACKOFF_CAP_MS = 15_000;
/** From this attempt on, each retry first asks the session whether the turn
 *  is still running: a network that keeps failing is also a turn that may
 *  have ended without us. */
const POLL_FROM_ATTEMPT = 4;

export type LifecycleKind = "idle" | "streaming" | "reconnecting" | "settled";

export interface Lifecycle {
  kind: LifecycleKind;
  /** Connections lost since the stream last carried a frame. */
  attempt: number;
  /** Once settled: the streamed rows match the database's, so no hydrate. */
  verified: boolean | null;
}

export type LifecycleInput =
  | { type: "stream-opened" }
  | { type: "frames-flowing" }
  | { type: "connection-lost"; failures: number }
  | { type: "heartbeat-missed" }
  | { type: "tab-visible"; hiddenMs: number; sinceLastFrameMs: number }
  | { type: "finished"; verified: boolean }
  | { type: "closed" };

export type LifecycleEffect =
  | { type: "reconnect"; delayMs: number; pollFirst: boolean }
  | { type: "wait-for-visible" }
  | { type: "probe-session"; reason: "finish" | "wake" }
  | null;

export const initialLifecycle: Lifecycle = {
  kind: "idle",
  attempt: 0,
  verified: null,
};

export function transition(
  state: Lifecycle,
  input: LifecycleInput,
  {
    visible,
    random = Math.random,
  }: { visible: boolean; random?: () => number },
): { state: Lifecycle; effect: LifecycleEffect } {
  switch (input.type) {
    case "stream-opened":
      return { state: streaming(), effect: null };
    case "frames-flowing":
      return state.kind === "reconnecting"
        ? { state: streaming(), effect: null }
        : { state, effect: null };
    case "connection-lost":
      return lose(Math.max(1, input.failures), visible, random);
    case "heartbeat-missed":
      if (state.kind !== "streaming") return { state, effect: null };
      return {
        state: { ...state, kind: "reconnecting", attempt: state.attempt + 1 },
        effect: { type: "reconnect", delayMs: 0, pollFirst: false },
      };
    case "tab-visible":
      return wake(state, input);
    case "finished":
      return {
        state: { kind: "settled", attempt: 0, verified: input.verified },
        effect: { type: "probe-session", reason: "finish" },
      };
    case "closed":
      return { state: initialLifecycle, effect: null };
  }
}

/** Exponential, capped, with jitter so a fleet of tabs does not reconnect in step. */
export function backoffDelay(attempt: number, random: () => number) {
  const ceiling = Math.min(
    BACKOFF_CAP_MS,
    BACKOFF_BASE_MS * 2 ** Math.max(0, attempt - 1),
  );
  return Math.round(ceiling * (0.5 + random() * 0.5));
}

function streaming(): Lifecycle {
  return { kind: "streaming", attempt: 0, verified: null };
}

function lose(attempt: number, visible: boolean, random: () => number) {
  const next: Lifecycle = { kind: "reconnecting", attempt, verified: null };
  if (!visible) {
    return { state: next, effect: { type: "wait-for-visible" } as const };
  }
  return {
    state: next,
    effect: {
      type: "reconnect",
      delayMs: backoffDelay(attempt, random),
      pollFirst: attempt >= POLL_FROM_ATTEMPT,
    } as const,
  };
}

function wake(
  state: Lifecycle,
  input: Extract<LifecycleInput, { type: "tab-visible" }>,
): { state: Lifecycle; effect: LifecycleEffect } {
  if (state.kind === "reconnecting") {
    return {
      state: { ...state, attempt: 0 },
      effect: { type: "reconnect", delayMs: 0, pollFirst: false },
    };
  }
  if (state.kind === "streaming") {
    if (input.sinceLastFrameMs < HEARTBEAT_MISS_MS) {
      return { state, effect: null };
    }
    return {
      state: { ...state, kind: "reconnecting", attempt: 1 },
      effect: { type: "reconnect", delayMs: 0, pollFirst: false },
    };
  }
  if (input.hiddenMs < WAKE_RESYNC_THRESHOLD_MS) {
    return { state, effect: null };
  }
  return { state, effect: { type: "probe-session", reason: "wake" } };
}
