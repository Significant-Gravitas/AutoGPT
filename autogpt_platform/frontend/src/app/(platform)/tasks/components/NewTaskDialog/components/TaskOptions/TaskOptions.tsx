"use client";

import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import type { TaskOptionsState } from "../../useTaskOptions";
import { RepeatPicker } from "./RepeatPicker";
import { StartPicker } from "./StartPicker";

interface Props {
  state: TaskOptionsState;
}

/** When the task starts or how often it runs (one or the other), what
 *  done looks like and what it is for. Each refusal from the server shows
 *  under the part it is about. */
export function TaskOptions({ state }: Props) {
  const { options, refusal } = state;

  return (
    <div
      role="group"
      aria-label="Task options"
      className="flex flex-col gap-2 border-t border-border-base px-1 pt-2"
    >
      <div className="flex flex-wrap items-center gap-2">
        <StartPicker
          key={options.repeat ? "repeating" : "once"}
          value={options.startAt}
          onChange={state.setStart}
        />
        <RepeatPicker value={options.repeat} onChange={state.setRepeat} />
      </div>
      {refusal && refusal.field !== "form" ? (
        <Refusal text={refusal.text} />
      ) : null}
      <div className="grid gap-2 sm:grid-cols-2">
        <Input
          id="new-task-done-when"
          label="Done when"
          hideLabel
          size="small"
          placeholder="Done when…"
          value={options.doneWhen}
          onChange={(e) => state.setDoneWhen(e.target.value)}
          wrapperClassName="mb-0"
        />
        <Input
          id="new-task-goal"
          label="Goal"
          hideLabel
          size="small"
          placeholder="Goal"
          value={options.goal}
          onChange={(e) => state.setGoal(e.target.value)}
          wrapperClassName="mb-0"
        />
      </div>
    </div>
  );
}

export function Refusal({ text }: { text: string }) {
  return (
    <Text as="p" variant="small" tone="error" role="alert">
      {text}
    </Text>
  );
}
