import { ControlValidity } from "../ControlValidity";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { TaskBoard } from "@/lib/openui/catalog-tasks";
import { Button } from "@/components/atoms/Button/Button";
import { useOpenUIDisabled } from "../../interactionContext";
import { TaskCard } from "./TaskCard";
import { useTaskBoard } from "./useTaskBoard";

const columns = [
  { id: "todo", label: "To do" },
  { id: "doing", label: "Doing" },
  { id: "done", label: "Done" },
];
export function TaskBoardView({
  props,
}: ComponentRenderProps<z.infer<typeof TaskBoard.props>>) {
  const board = useTaskBoard(props);
  const disabled = useOpenUIDisabled();
  return (
    <fieldset disabled={disabled} className="min-w-0 space-y-3">
      <legend className="text-sm font-semibold">{props.title}</legend>
      <p className="text-xs text-muted-foreground">
        Local planning and reported progress
      </p>
      <ControlValidity label={props.title} error={board.error} />
      {board.actionError && (
        <p role="alert" className="text-sm text-destructive">
          {board.actionError}
        </p>
      )}
      <div className="grid gap-3 lg:grid-cols-3">
        {columns.map((column) => (
          <section
            key={column.id}
            className="min-w-0 space-y-3 rounded-xl bg-muted/40 p-3"
            aria-label={column.label}
          >
            <h3 className="text-sm font-semibold">
              {column.label} ·{" "}
              {board.tasks.filter((task) => task.status === column.id).length}
            </h3>
            {board.tasks
              .filter((task) => task.status === column.id)
              .map((task) => (
                <TaskCard
                  key={task.id}
                  task={task}
                  tasks={board.tasks}
                  disabled={disabled}
                  removable={props.allowAdd}
                  onChange={board.change}
                  onRemove={() => board.remove(task.id)}
                />
              ))}
          </section>
        ))}
      </div>
      {props.allowAdd && (
        <Button
          type="button"
          variant="secondary"
          size="small"
          onClick={board.add}
          disabled={board.tasks.length >= 30}
        >
          Add task
        </Button>
      )}
    </fieldset>
  );
}
