import { useRef, useState } from "react";

export type SheetView = "summary" | "changes" | "sources" | "history";

export interface SkillEditDraft {
  body: string;
  description: string;
  triggers: string[];
  keepAutoImprove: boolean;
  baseVersionId: string | null;
}

export interface SkillReviewState {
  view: SheetView;
  selectedVersionId: string | null;
  editor: SkillEditDraft | null;
  decisions: Record<string, string>;
}

export type UpdateSkillReview = (
  update:
    | Partial<SkillReviewState>
    | ((previous: SkillReviewState) => Partial<SkillReviewState>),
) => void;

export function useSkillReviewState(
  expertId: string | null,
  skillName: string | null,
  initialVersionId: string | null,
) {
  const [records, setRecords] = useState<Record<string, SkillReviewState>>({});
  const scrollPositions = useRef(new Map<string, number>());
  const key = JSON.stringify([expertId, skillName, initialVersionId]);
  const initial: SkillReviewState = {
    view: "summary",
    selectedVersionId: initialVersionId,
    editor: null,
    decisions: {},
  };
  const state = records[key] ?? initial;

  function update(change: Parameters<UpdateSkillReview>[0]) {
    setRecords((previous) => {
      const current = previous[key] ?? initial;
      const patch = typeof change === "function" ? change(current) : change;
      return { ...previous, [key]: { ...current, ...patch } };
    });
  }

  function restoreScroll(element: HTMLDivElement | null) {
    if (element) element.scrollTop = scrollPositions.current.get(key) ?? 0;
  }

  function rememberScroll(top: number) {
    scrollPositions.current.set(key, top);
  }

  return { state, update, restoreScroll, rememberScroll };
}
