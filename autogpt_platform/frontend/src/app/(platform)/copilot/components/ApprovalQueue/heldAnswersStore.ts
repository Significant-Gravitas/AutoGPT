import { create } from "zustand";

export type HeldAnswer = "approved" | "rejected";

// Answers given on this page, by review id, so a held call's row shows the
// answer at the click; the late result that the continuation turn persists
// supersedes it.
interface HeldAnswersState {
  answers: Readonly<Record<string, HeldAnswer>>;
  record(reviewIds: string[], approved: boolean): void;
}

export const useHeldAnswersStore = create<HeldAnswersState>((set) => ({
  answers: {},
  record: (reviewIds, approved) =>
    set((state) => ({
      answers: {
        ...state.answers,
        ...Object.fromEntries(
          reviewIds.map((id) => [id, approved ? "approved" : "rejected"]),
        ),
      },
    })),
}));
