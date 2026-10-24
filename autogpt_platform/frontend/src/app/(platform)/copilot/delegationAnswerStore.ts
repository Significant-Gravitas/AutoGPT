import { create } from "zustand";

export interface DelegationAnswer {
  /** The question the answer was for, so a new question re-opens the card. */
  question: string | null;
  text: string;
  sentAt: number;
}

interface DelegationAnswerState {
  answers: Record<string, DelegationAnswer>;
  recordAnswer: (subSessionId: string, answer: DelegationAnswer) => void;
  clearAnswer: (subSessionId: string) => void;
}

/** Answers the user sent to a teammate's question from Otto's chat. Held
 *  in memory only: it bridges the gap until the teammate's own session
 *  shows it resumed, and a reload reads the truth off that session. */
export const useDelegationAnswerStore = create<DelegationAnswerState>(
  (set) => ({
    answers: {},
    recordAnswer: (subSessionId, answer) =>
      set((state) => ({
        answers: { ...state.answers, [subSessionId]: answer },
      })),
    clearAnswer: (subSessionId) =>
      set((state) => {
        if (!(subSessionId in state.answers)) return state;
        const { [subSessionId]: _dropped, ...rest } = state.answers;
        return { answers: rest };
      }),
  }),
);
