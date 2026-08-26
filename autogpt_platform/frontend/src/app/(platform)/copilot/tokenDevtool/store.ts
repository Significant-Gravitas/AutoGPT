import { create } from "zustand";
import {
  computeBreakdown,
  type ContextBreakdown,
  type MessageLike,
  type TokenTurn,
} from "./tokenMath";

/** Display cap for the per-turn list in the popover. The context estimate is
 *  kept separately precisely so this cap cannot silently drop turns it
 *  depends on. */
const KEPT_TURNS = 50;

/** Turns are capped per session; without this the session keys themselves
 *  would grow without bound as a long-lived tab visits more threads. */
const KEPT_SESSIONS = 20;

function withSession<T>(
  entries: Record<string, T>,
  sessionId: string,
  value: T,
): Record<string, T> {
  const next = { ...entries, [sessionId]: value };
  const keys = Object.keys(next);
  if (keys.length <= KEPT_SESSIONS) return next;
  const dropped = new Set(
    keys
      .filter((key) => key !== sessionId)
      .slice(0, keys.length - KEPT_SESSIONS),
  );
  return Object.fromEntries(
    Object.entries(next).filter(([key]) => !dropped.has(key)),
  );
}

interface TokenDevtoolState {
  turnsBySession: Record<string, TokenTurn[]>;
  breakdownBySession: Record<string, ContextBreakdown>;
  /** Running cache-write sum since the session's last compaction. */
  liveContextBySession: Record<string, number>;
  /** Sticky: once a session has compacted, the loaded-history seed is stale
   *  for good — it must not un-stick when the compaction turn scrolls out of
   *  the kept window. */
  compactedBySession: Record<string, boolean>;
  record: (sessionId: string, turn: TokenTurn) => void;
  setBreakdown: (sessionId: string, breakdown: ContextBreakdown) => void;
}

export const useTokenDevtoolStore = create<TokenDevtoolState>((set) => ({
  turnsBySession: {},
  breakdownBySession: {},
  liveContextBySession: {},
  compactedBySession: {},
  record(sessionId, turn) {
    set((state) => ({
      turnsBySession: withSession(
        state.turnsBySession,
        sessionId,
        [...(state.turnsBySession[sessionId] ?? []), turn].slice(-KEPT_TURNS),
      ),
      // A compaction turn restarts the sum: the compacted context is
      // re-written to cache in that same turn.
      liveContextBySession: withSession(
        state.liveContextBySession,
        sessionId,
        turn.compacted
          ? turn.cacheCreationTokens
          : (state.liveContextBySession[sessionId] ?? 0) +
              turn.cacheCreationTokens,
      ),
      compactedBySession: withSession(
        state.compactedBySession,
        sessionId,
        turn.compacted || (state.compactedBySession[sessionId] ?? false),
      ),
    }));
  },
  setBreakdown(sessionId, breakdown) {
    set((state) => ({
      breakdownBySession: withSession(
        state.breakdownBySession,
        sessionId,
        breakdown,
      ),
    }));
  },
}));

/** Recompute the session's history breakdown from the loaded messages. */
export function updateHistoryBreakdown(
  sessionId: string,
  messages: readonly MessageLike[],
) {
  useTokenDevtoolStore
    .getState()
    .setBreakdown(sessionId, computeBreakdown(messages));
}
