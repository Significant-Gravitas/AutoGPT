import type { TaskDraft } from "@/app/(platform)/tasks/components/NewTaskDialog/draft";
import { OTTO_VALUE } from "@/app/(platform)/tasks/components/NewTaskDialog/helpers";
import { NO_OPTIONS } from "@/app/(platform)/tasks/components/NewTaskDialog/taskOptions";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";

/** At most this many suggestions on the empty board. */
export const FIRST_TASKS = 3;

/** The suggestions the user has not started or dismissed, first few. */
export function openSuggestions(tasks: SuggestedTask[]) {
  return tasks
    .filter((task) => (task.state ?? "open") === "open")
    .slice(0, FIRST_TASKS);
}

/** A suggestion as a draft for the create dialog, with its expert when
 *  that expert is on the team. */
export function draftFromSuggestion(
  suggestion: SuggestedTask,
  experts: Expert[],
): TaskDraft {
  const prompt = suggestion.prompt.trim();
  const owner = experts.find((expert) => expert.id === suggestion.expert_id);
  return {
    text:
      prompt && prompt !== suggestion.title
        ? `${suggestion.title}\n${prompt}`
        : suggestion.title,
    owner: owner?.id ?? OTTO_VALUE,
    options: NO_OPTIONS,
  };
}
