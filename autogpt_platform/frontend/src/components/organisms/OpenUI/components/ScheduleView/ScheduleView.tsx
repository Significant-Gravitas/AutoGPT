import type { ComponentRenderProps } from "@openuidev/react-lang";
import {
  calculateSchedule,
  readScheduleTasks,
  scheduleTime,
  type ScheduleProps,
  type ScheduleDraft,
} from "@/lib/openui/schedule";
import { useOpenUIDisabled } from "../../interactionContext";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";
import { ScheduleTaskRow } from "./ScheduleTaskRow";

export function ScheduleView({ props }: ComponentRenderProps<ScheduleProps>) {
  const disabled = useOpenUIDisabled();
  const field = useFieldView<ScheduleDraft[]>(props.name, props.tasks ?? []);
  const tasks = readScheduleTasks(field.value, props.tasks ?? []);
  const result = props.days?.length
    ? calculateSchedule(props.days, tasks)
    : null;
  useDerivedField(`${props.name}_required_minutes`, result?.required ?? null);
  useDerivedField(`${props.name}_capacity_minutes`, result?.capacity ?? null);
  useDerivedField(`${props.name}_buffer_minutes`, result?.buffer ?? null);
  useDerivedField(
    `${props.name}_unscheduled_minutes`,
    result?.unscheduled ?? null,
  );
  useDerivedField(`${props.name}_feasible`, result?.feasible ?? false);

  function move(index: number, direction: number) {
    const changed = [...tasks];
    [changed[index], changed[index + direction]] = [
      changed[index + direction],
      changed[index],
    ];
    field.setValue(changed);
  }

  return (
    <fieldset disabled={disabled} className="min-w-0 space-y-3">
      <legend className="text-sm font-semibold text-foreground">
        {props.title}
      </legend>
      <p className="text-xs text-muted-foreground">
        Edit durations, order or completion. Available times stay fixed; send a
        chat follow-up to change them.
      </p>
      <ul
        aria-label="Available times"
        className="text-sm text-muted-foreground"
      >
        {props.days?.map((day) => (
          <li key={day.label}>
            {day.label}:{" "}
            {day.windows
              .map((window) => `${window.start}–${window.end}`)
              .join(", ")}
          </li>
        ))}
      </ul>
      <div
        role="status"
        aria-live="polite"
        className="space-y-1 rounded-lg bg-muted p-3 text-sm text-foreground"
      >
        {result ? (
          <>
            <p>
              {result.required} min remaining · {result.capacity} min available
            </p>
            {result.buffer < 0 && (
              <p>{-result.buffer} min over available time</p>
            )}
            <p>
              {result.feasible
                ? `${result.buffer} min spare`
                : `${result.unscheduled} min unscheduled · ${result.unused} min unused`}
            </p>
            {!result.feasible && (
              <p className="text-destructive">
                This plan does not fit. Change durations, order or availability
                before treating it as feasible.
              </p>
            )}
          </>
        ) : (
          <p>Enter valid durations to calculate the schedule.</p>
        )}
      </div>
      <ol className="space-y-3">
        {tasks.map((task, index) => {
          const allocation = result?.allocations[index];
          return (
            <ScheduleTaskRow
              key={task.id}
              task={task}
              disabled={disabled}
              first={index === 0}
              last={index === tasks.length - 1}
              times={
                allocation?.slots.map(
                  (slot) =>
                    `${scheduleTime(slot.start, props.days)}–${scheduleTime(slot.end, props.days)}`,
                ) ?? []
              }
              conflict={allocation?.conflict ?? ""}
              onMove={(direction) => move(index, direction)}
              onChange={(changed) =>
                field.setValue(
                  tasks.map((item) =>
                    item.id === changed.id ? changed : item,
                  ),
                )
              }
            />
          );
        })}
      </ol>
    </fieldset>
  );
}
