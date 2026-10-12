import {
  resolveExpertIdentity,
  useExpertMap,
} from "@/app/(platform)/copilot/useExpertMap";
import { shortAge } from "../NeedsYou/helpers";
import { type Card, cardState, stateLineText } from "../../board-helpers";
import { OTTO_OPTION } from "../AssigneePicker/helpers";

export function useTaskCard(card: Card) {
  const {
    isExpertsEnabled,
    expertsById,
    hasExpertsSettled,
    hasExpertsErrored,
  } = useExpertMap();
  const state = cardState(card);
  const expertId = card.expert_id ?? null;
  const expert = resolveExpertIdentity(expertId, expertsById, {
    settled: !isExpertsEnabled || hasExpertsSettled,
    errored: hasExpertsErrored,
  });

  return {
    title: card.title || "Untitled task",
    stateLine: stateLineText(state),
    isRunning: state.kind === "running",
    needsYou: !!card.needs_you,
    fileCount: card.file_count ?? 0,
    age: shortAge(card.updated_at),
    assignee: expertId
      ? {
          name: expert?.name ?? null,
          avatarUrl: expert?.avatarUrl ?? null,
          color: expert?.color ?? null,
          isAutopilot: false,
        }
      : {
          name: OTTO_OPTION.name,
          avatarUrl: null,
          color: null,
          isAutopilot: true,
        },
  };
}
