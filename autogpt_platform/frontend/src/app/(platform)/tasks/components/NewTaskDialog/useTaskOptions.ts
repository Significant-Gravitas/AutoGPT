import { useState } from "react";
import {
  fieldOfRefusal,
  hasOptions,
  NO_OPTIONS,
  refusalText,
  type OptionsField,
  type TaskOptions,
} from "./taskOptions";
import type { RepeatChoice } from "./repeat";
import type { StartChoice } from "./startChoice";

interface Refusal {
  field: OptionsField;
  text: string;
}

/** The create dialog's options row: folded away until asked for. A start
 *  time and a rhythm replace each other. */
export function useTaskOptions(initial: TaskOptions = NO_OPTIONS) {
  const [isOpen, setIsOpen] = useState(() => hasOptions(initial));
  const [options, setOptions] = useState<TaskOptions>(initial);
  const [refusal, setRefusal] = useState<Refusal | null>(null);

  function change(next: Partial<TaskOptions>) {
    setOptions((current) => ({ ...current, ...next }));
    setRefusal(null);
  }

  return {
    isOpen,
    toggle: () => setIsOpen((open) => !open),
    options,
    setStart: (start: StartChoice | null) =>
      change(start ? { start, repeat: null } : { start }),
    setRepeat: (repeat: RepeatChoice | null) =>
      change(repeat ? { repeat, start: null } : { repeat }),
    setDoneWhen: (doneWhen: string) => change({ doneWhen }),
    setGoal: (goal: string) => change({ goal }),
    /** The server's sentence, shown under the part it is about. */
    refusal,
    refuse(message: string) {
      const field = fieldOfRefusal(message);
      setRefusal({ field, text: refusalText(message) });
      if (field !== "form") setIsOpen(true);
    },
    reset() {
      setIsOpen(false);
      setOptions(NO_OPTIONS);
      setRefusal(null);
    },
  };
}

export type TaskOptionsState = ReturnType<typeof useTaskOptions>;
