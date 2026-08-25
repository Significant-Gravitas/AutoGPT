import { create } from "zustand";
import {
  computeBreakdown,
  type ContextBreakdown,
  type TokenTurn,
} from "./tokenMath";

const KEPT_TURNS = 50;

interface TokenDevtoolState {
  turnsBySession: Record<string, TokenTurn[]>;
  breakdownBySession: Record<string, ContextBreakdown>;
  record: (sessionId: string, turn: TokenTurn) => void;
  setBreakdown: (sessionId: string, breakdown: ContextBreakdown) => void;
}

export const useTokenDevtoolStore = create<TokenDevtoolState>((set) => ({
  turnsBySession: {},
  breakdownBySession: {},
  record(sessionId, turn) {
    set((state) => ({
      turnsBySession: {
        ...state.turnsBySession,
        [sessionId]: [...(state.turnsBySession[sessionId] ?? []), turn].slice(
          -KEPT_TURNS,
        ),
      },
    }));
  },
  setBreakdown(sessionId, breakdown) {
    set((state) => ({
      breakdownBySession: {
        ...state.breakdownBySession,
        [sessionId]: breakdown,
      },
    }));
  },
}));

/** Recompute the session's history breakdown from the loaded messages. */
export function updateHistoryBreakdown(sessionId: string, messages: unknown[]) {
  useTokenDevtoolStore
    .getState()
    .setBreakdown(sessionId, computeBreakdown(messages));
}
