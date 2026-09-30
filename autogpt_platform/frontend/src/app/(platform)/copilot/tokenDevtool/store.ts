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

interface TokenDevtoolState {
  turnsBySession: Record<string, TokenTurn[]>;
  breakdownBySession: Record<string, ContextBreakdown>;
  /** Running cache-write sum since the session's last compaction. */
  liveContextBySession: Record<string, number>;
  /** Sticky: once a session has compacted, the loaded-history seed is stale
   *  for good — it must not un-stick when the compaction turn scrolls out of
   *  the kept window. */
  compactedBySession: Record<string, boolean>;
  /** Least-recently-touched first. Single source of truth for retention, so
   *  the maps cannot prune to different key sets. */
  sessionOrder: string[];
  record: (sessionId: string, turn: TokenTurn) => void;
  setBreakdown: (sessionId: string, breakdown: ContextBreakdown) => void;
}

/** Moves sessionId to most-recent and reports which sessions fall off the
 *  end. Every map prunes against this one answer. */
function touchSession(order: string[], sessionId: string) {
  const next = [...order.filter((id) => id !== sessionId), sessionId];
  if (next.length <= KEPT_SESSIONS) return { order: next, dropped: EMPTY };
  return {
    order: next.slice(-KEPT_SESSIONS),
    dropped: new Set(next.slice(0, next.length - KEPT_SESSIONS)),
  };
}

const EMPTY: ReadonlySet<string> = new Set();

function without<T>(
  entries: Record<string, T>,
  dropped: ReadonlySet<string>,
): Record<string, T> {
  if (dropped.size === 0) return entries;
  return Object.fromEntries(
    Object.entries(entries).filter(([key]) => !dropped.has(key)),
  );
}

export const useTokenDevtoolStore = create<TokenDevtoolState>((set) => ({
  turnsBySession: {},
  breakdownBySession: {},
  liveContextBySession: {},
  compactedBySession: {},
  sessionOrder: [],
  record(sessionId, turn) {
    set((state) => {
      const { order, dropped } = touchSession(state.sessionOrder, sessionId);
      return {
        sessionOrder: order,
        breakdownBySession: without(state.breakdownBySession, dropped),
        turnsBySession: {
          ...without(state.turnsBySession, dropped),
          [sessionId]: [...(state.turnsBySession[sessionId] ?? []), turn].slice(
            -KEPT_TURNS,
          ),
        },
        // A compaction turn restarts the sum: the compacted context is
        // re-written to cache in that same turn.
        liveContextBySession: {
          ...without(state.liveContextBySession, dropped),
          [sessionId]: turn.compacted
            ? turn.cacheCreationTokens
            : (state.liveContextBySession[sessionId] ?? 0) +
              turn.cacheCreationTokens,
        },
        compactedBySession: {
          ...without(state.compactedBySession, dropped),
          [sessionId]:
            turn.compacted || (state.compactedBySession[sessionId] ?? false),
        },
      };
    });
  },
  setBreakdown(sessionId, breakdown) {
    set((state) => {
      const { order, dropped } = touchSession(state.sessionOrder, sessionId);
      return {
        sessionOrder: order,
        turnsBySession: without(state.turnsBySession, dropped),
        liveContextBySession: without(state.liveContextBySession, dropped),
        compactedBySession: without(state.compactedBySession, dropped),
        breakdownBySession: {
          ...without(state.breakdownBySession, dropped),
          [sessionId]: breakdown,
        },
      };
    });
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
