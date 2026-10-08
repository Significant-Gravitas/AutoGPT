import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import type { getOrCreateCopilotChatRuntime } from "./copilotChatRegistry";
import { getActiveBackendTurnId } from "./helpers";
import {
  HEARTBEAT_INTERVAL_MS,
  HEARTBEAT_MISS_MS,
  initialLifecycle,
  transition,
  type Lifecycle,
  type LifecycleEffect,
  type LifecycleInput,
} from "./stream/lifecycle";
import type { TurnStream } from "./stream/turnStream";

// A turn the backend starts on its own (a held call's continuation, an
// engine switch, an answered approval) is dispatched just as the previous one
// ends or the POST returns, so the session is probed a few times for it.
const PROBE_INTERVAL_MS = 500;
const FINISH_PROBE_ATTEMPTS = 1;
const FINISH_PROBE_ATTEMPTS_PENDING_SWITCH = 8;
const FOLLOW_PROBE_ATTEMPTS = 8;

type ChatStatus = "submitted" | "streaming" | "ready" | "error";
type SessionResult = { data?: unknown; isError?: boolean };

interface Args {
  runtime: ReturnType<typeof getOrCreateCopilotChatRuntime> | null;
  status: ChatStatus;
  hasActiveStream: boolean;
  /** The turn the session view reports running; null when it names none. */
  activeTurnId: string | null;
  /** The hydrated rows are in the chat, so a resume trims against them. */
  canAttach: boolean;
  refetchSession: () => Promise<SessionResult>;
  /** Start reading `turnId` with a fresh parser (`resumeStream`). */
  attachTurn: (turnId: string | null) => void;
  /** Whether the finished turn switched engines; a switch settles slower. */
  consumeEngineSwitch: () => boolean;
}

/**
 * Runs the stream lifecycle (`stream/lifecycle.ts`) for one chat: the only
 * place that reconnects, resumes or follows a turn. Inputs are the turn
 * stream's own state (connection lost, finished), missed heartbeats, the tab
 * becoming visible again, and the session view naming a turn this chat has
 * not read yet.
 */
export function useCopilotStreamLifecycle(args: Args) {
  const { runtime, status, hasActiveStream, activeTurnId, canAttach } = args;
  const transport = runtime?.transport ?? null;
  const streamState = useSyncExternalStore(
    transport?.subscribe ?? subscribeNowhere,
    () => transport?.activeStream?.getState() ?? null,
    () => null,
  );
  const [lifecycle, setLifecycle] = useState<Lifecycle>(initialLifecycle);
  const [isFinishProbing, setIsFinishProbing] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const lifecycleRef = useRef(lifecycle);
  const argsRef = useRef(args);
  argsRef.current = args;
  const seenTurnsRef = useRef(new Set<string>());
  const attachedUnnamedRef = useRef(false);
  const reconnectTimerRef = useRef<number | undefined>(undefined);
  const probeRef = useRef({ budget: 0, running: false });
  const pendingFollowRef = useRef(false);
  const mountedRef = useRef(true);
  const isBusy = status === "streaming" || status === "submitted";

  function dispatch(input: LifecycleInput) {
    const { state, effect } = transition(lifecycleRef.current, input, {
      visible: document.visibilityState !== "hidden",
    });
    lifecycleRef.current = state;
    if (mountedRef.current) setLifecycle(state);
    runEffect(effect);
  }

  function runEffect(effect: LifecycleEffect) {
    if (!effect) return;
    if (effect.type === "probe-session") {
      if (effect.reason === "wake") void wakeSync();
      else void probeForTurn({ attempts: finishProbeAttempts(), finish: true });
      return;
    }
    window.clearTimeout(reconnectTimerRef.current);
    if (effect.type === "wait-for-visible") return;
    reconnectTimerRef.current = window.setTimeout(
      () => void reconnect(effect.pollFirst),
      effect.delayMs,
    );
  }

  async function reconnect(pollFirst: boolean) {
    const stream = transport?.activeStream;
    if (!stream || !mountedRef.current) return;
    if (pollFirst) {
      const running = await activeTurnFromSession();
      const turnId = stream.getState().turnId;
      if (running === null || (running && turnId && running !== turnId)) {
        stream.abandon();
        return;
      }
    }
    stream.reconnect();
  }

  async function activeTurnFromSession() {
    try {
      const result = await argsRef.current.refetchSession();
      return result.isError ? undefined : getActiveBackendTurnId(result);
    } catch {
      return undefined;
    }
  }

  function finishProbeAttempts() {
    return argsRef.current.consumeEngineSwitch()
      ? FINISH_PROBE_ATTEMPTS_PENDING_SWITCH
      : FINISH_PROBE_ATTEMPTS;
  }

  // Refetch the session until it names a turn this chat has not read (the
  // attach effect below picks it up) or the budget runs out.
  async function probeForTurn({
    attempts,
    finish,
  }: {
    attempts: number;
    finish: boolean;
  }) {
    const probe = probeRef.current;
    probe.budget = attempts;
    if (finish) setIsFinishProbing(true);
    if (probe.running) return;
    probe.running = true;
    try {
      while (probe.budget-- > 0) {
        await new Promise((wake) => setTimeout(wake, PROBE_INTERVAL_MS));
        if (!mountedRef.current) return;
        const running = await activeTurnFromSession();
        if (!mountedRef.current) return;
        if (running && !seenTurnsRef.current.has(running)) return;
      }
    } finally {
      probe.running = false;
      if (mountedRef.current) setIsFinishProbing(false);
    }
  }

  async function wakeSync() {
    setIsSyncing(true);
    await activeTurnFromSession();
    if (mountedRef.current) setIsSyncing(false);
  }

  function onStreamChange(stream: TurnStream, isNew: boolean) {
    const state = stream.getState();
    if (state.turnId) seenTurnsRef.current.add(state.turnId);
    switch (state.phase) {
      case "connecting":
        if (isNew) dispatch({ type: "stream-opened" });
        return;
      case "open":
        dispatch({ type: isNew ? "stream-opened" : "frames-flowing" });
        return;
      case "lost":
        if (isNew) dispatch({ type: "stream-opened" });
        dispatch({ type: "connection-lost", failures: state.failures });
        return;
      case "finished":
        if (!isNew) dispatch({ type: "finished", verified: !!state.verified });
        return;
      case "closed":
        if (!isNew) dispatch({ type: "closed" });
        return;
    }
  }

  useEffect(() => {
    mountedRef.current = true;
    if (!transport) return;
    let current: TurnStream | null = null;
    let last: unknown = null;
    function check() {
      const stream = transport?.activeStream ?? null;
      if (!stream) return;
      const state = stream.getState();
      if (stream === current && state === last) return;
      const isNew = stream !== current;
      current = stream;
      last = state;
      onStreamChange(stream, isNew);
    }
    check();
    const unsubscribe = transport.subscribe(check);
    return () => {
      unsubscribe();
      mountedRef.current = false;
      window.clearTimeout(reconnectTimerRef.current);
    };
  }, [transport]);

  // The watchdog fires on missing heartbeats only: a model or tool that is
  // silent for minutes still has a stream carrying one every 10 s.
  useEffect(() => {
    if (lifecycle.kind !== "streaming" || !transport) return;
    const id = window.setInterval(() => {
      const stream = transport.activeStream;
      const phase = stream?.getState().phase;
      if (!stream || (phase !== "open" && phase !== "connecting")) return;
      if (Date.now() - stream.lastFrameAt >= HEARTBEAT_MISS_MS) {
        dispatch({ type: "heartbeat-missed" });
      }
    }, HEARTBEAT_INTERVAL_MS / 2);
    return () => window.clearInterval(id);
  }, [lifecycle.kind, transport]);

  useEffect(() => {
    let hiddenAt = Date.now();
    function onVisibilityChange() {
      const now = Date.now();
      if (document.visibilityState === "hidden") {
        hiddenAt = now;
        return;
      }
      const stream = transport?.activeStream;
      dispatch({
        type: "tab-visible",
        hiddenMs: now - hiddenAt,
        sinceLastFrameMs: stream ? now - stream.lastFrameAt : Infinity,
      });
    }
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () =>
      document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [transport]);

  // A turn the session view names and this chat has not read: a reload into
  // a running turn, a chained continuation, an answered approval. Deferred a
  // tick so this commit's hydrate lands first and the resume trims against it.
  useEffect(() => {
    if (!runtime || !canAttach || !hasActiveStream || isBusy) return;
    const id = window.setTimeout(() => {
      if (!mountedRef.current || !shouldAttach(runtime, activeTurnId)) return;
      if (activeTurnId) seenTurnsRef.current.add(activeTurnId);
      else attachedUnnamedRef.current = true;
      argsRef.current.attachTurn(activeTurnId);
    }, 0);
    return () => window.clearTimeout(id);
  }, [runtime, canAttach, hasActiveStream, activeTurnId, isBusy, lifecycle]);

  function shouldAttach(
    chat: NonNullable<Args["runtime"]>,
    turnId: string | null,
  ) {
    if (lifecycleRef.current.kind === "reconnecting") return false;
    const phase = chat.transport.activeStream?.getState().phase;
    if (phase && phase !== "finished" && phase !== "closed") return false;
    const stopped = chat.stoppedTurnId;
    if (!turnId) return !attachedUnnamedRef.current && stopped === null;
    if (seenTurnsRef.current.has(turnId)) return false;
    return stopped === null || (stopped !== "" && stopped !== turnId);
  }

  useEffect(() => {
    if (isBusy || !pendingFollowRef.current) return;
    pendingFollowRef.current = false;
    void probeForTurn({ attempts: FOLLOW_PROBE_ATTEMPTS, finish: false });
  }, [isBusy]);

  /** An answered approval card starts the chat's next turn on the server. */
  function followBackendTurn() {
    if (runtime) runtime.stoppedTurnId = null;
    attachedUnnamedRef.current = false;
    pendingFollowRef.current = true;
    if (isBusy) return;
    pendingFollowRef.current = false;
    void probeForTurn({ attempts: FOLLOW_PROBE_ATTEMPTS, finish: false });
  }

  /** The Stop button: close the connection and never resume this turn. */
  function stopTurn() {
    if (!runtime) return;
    const stream = runtime.transport.activeStream;
    runtime.stoppedTurnId = stream?.getState().turnId ?? activeTurnId ?? "";
    window.clearTimeout(reconnectTimerRef.current);
    stream?.close();
  }

  /** A send starts a new turn: a stopped one no longer blocks resumes. */
  function clearStop() {
    if (runtime) runtime.stoppedTurnId = null;
  }

  return {
    lifecycle,
    isReconnecting: lifecycle.kind === "reconnecting",
    isFinishProbing,
    isSyncing,
    /** Once a turn finished: whether its streamed rows match the database. */
    turnVerified:
      streamState?.phase === "finished" ? !!streamState.verified : null,
    followBackendTurn,
    stopTurn,
    clearStop,
  };
}

function subscribeNowhere() {
  return () => {};
}
