import {
  fromLocalInputValue,
  type QuickStart,
  toLocalInputValue,
} from "@/app/(platform)/tasks/schedule";
import { useTaskActions } from "@/app/(platform)/tasks/useTaskActions";
import type { Task } from "@/app/api/__generated__/models/task";
import { useState } from "react";

/** Re-time a scheduled task: a quick choice, a picked date and time, or no
 *  start time at all. A time already past is caught before it is sent. */
export function useScheduleEditor(task: Task, onDone: () => void) {
  const actions = useTaskActions(task.id);
  const [draft, setDraft] = useState(() => toLocalInputValue(task.start_at));
  const [error, setError] = useState<string | null>(null);

  function pick(choice: QuickStart) {
    actions.reschedule(choice.at(Date.now()), onDone);
  }

  function change(value: string | undefined) {
    setDraft(value ?? "");
    setError(null);
  }

  function save() {
    const at = fromLocalInputValue(draft);
    if (!at) return setError("Pick a date and time.");
    if (at.getTime() <= Date.now()) {
      return setError("Pick a time in the future.");
    }
    actions.reschedule(at, onDone);
  }

  return {
    draft,
    error,
    isSaving: actions.isPending,
    pick,
    change,
    save,
    remove: () => actions.unschedule(onDone),
  };
}
