"use client";

import { QUICK_STARTS } from "@/app/(platform)/tasks/schedule";
import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import { DateTimeInput } from "@/components/atoms/DateTimeInput/DateTimeInput";
import { Text } from "@/components/atoms/Text/Text";
import { useScheduleEditor } from "./useScheduleEditor";

interface Props {
  task: Task;
  onDone: () => void;
}

/** A few one-click start times, a date and time to pick, and a way to take
 *  the start time off so the task waits to be started by hand. */
export function ScheduleEditor({ task, onDone }: Props) {
  const editor = useScheduleEditor(task, onDone);

  return (
    <div
      role="group"
      aria-label="Change the start time"
      className="flex flex-col gap-2.5"
    >
      <div className="flex flex-wrap gap-1.5">
        {QUICK_STARTS.map((choice) => (
          <Button
            key={choice.label}
            variant="secondary"
            size="xs"
            disabled={editor.isSaving}
            onClick={() => editor.pick(choice)}
          >
            {choice.label}
          </Button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <DateTimeInput
          size="small"
          aria-label="Start time"
          value={editor.draft}
          onChange={editor.change}
          wrapperClassName="min-w-0 flex-1"
        />
        <Button
          variant="primary"
          size="xs"
          loading={editor.isSaving}
          disabled={!editor.draft}
          onClick={editor.save}
        >
          Set time
        </Button>
      </div>
      {editor.error ? (
        <Text as="p" variant="small" tone="error" role="alert">
          {editor.error}
        </Text>
      ) : null}
      <div className="flex items-center justify-between gap-2">
        <Button
          variant="ghost-muted"
          size="xs"
          disabled={editor.isSaving}
          onClick={editor.remove}
        >
          Remove start time
        </Button>
        <Button variant="ghost-muted" size="xs" onClick={onDone}>
          Close
        </Button>
      </div>
    </div>
  );
}
