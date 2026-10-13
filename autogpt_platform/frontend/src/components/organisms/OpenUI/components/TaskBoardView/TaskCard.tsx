import { useId } from "react";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import type { BoardTask } from "@/lib/openui/catalog-tasks";

interface Props {
  task: BoardTask;
  tasks: BoardTask[];
  disabled: boolean;
  removable: boolean;
  onChange: (task: BoardTask) => void;
  onRemove: () => void;
}

export function TaskCard({
  task,
  tasks,
  disabled,
  removable,
  onChange,
  onRemove,
}: Props) {
  const id = useId();
  const blocked = task.dependsOn.filter(
    (id) => tasks.find((item) => item.id === id)?.status !== "done",
  );
  return (
    <article className="space-y-2 rounded-lg border border-border bg-background p-3">
      <h4 className="break-words text-sm font-semibold">{task.title}</h4>
      <p className="text-xs text-muted-foreground">
        {task.owner || "Unassigned"}
        {task.due && ` · Due ${task.due}`}
      </p>
      {!!blocked.length && (
        <p className="text-warning text-xs">
          Waiting for{" "}
          {blocked
            .map((id) => tasks.find((item) => item.id === id)?.title)
            .join(", ")}
        </p>
      )}
      <label className="grid gap-1 text-xs">
        Status for {task.title}
        <select
          disabled={disabled}
          value={task.status}
          onChange={(event) =>
            onChange({
              ...task,
              status: event.target.value as BoardTask["status"],
            })
          }
          className="min-h-11 rounded-md border border-input bg-background px-2"
        >
          <option value="todo">To do</option>
          <option value="doing" disabled={!!blocked.length}>
            Doing
          </option>
          <option value="done" disabled={!!blocked.length}>
            Done
          </option>
        </select>
      </label>
      <details>
        <summary className="min-h-11 cursor-pointer py-3 text-xs">
          Edit {task.title}
        </summary>
        <div className="space-y-3">
          <Input
            id={`${id}-title`}
            label="Task title"
            value={task.title}
            required
            maxLength={200}
            onChange={(event) =>
              onChange({ ...task, title: event.target.value })
            }
          />
          <Input
            id={`${id}-owner`}
            label="Owner"
            value={task.owner}
            maxLength={120}
            onChange={(event) =>
              onChange({ ...task, owner: event.target.value })
            }
          />
          <Input
            id={`${id}-due`}
            label="Due date"
            type="date"
            value={task.due}
            onChange={(event) => onChange({ ...task, due: event.target.value })}
          />
          <fieldset className="space-y-1">
            <legend className="text-xs">Prerequisites</legend>
            {tasks
              .filter((item) => item.id !== task.id)
              .map((item) => (
                <label
                  key={item.id}
                  className="flex min-h-11 items-center gap-2 text-xs"
                >
                  <input
                    type="checkbox"
                    checked={task.dependsOn.includes(item.id)}
                    onChange={(event) =>
                      onChange({
                        ...task,
                        dependsOn: event.target.checked
                          ? [...task.dependsOn, item.id]
                          : task.dependsOn.filter((id) => id !== item.id),
                      })
                    }
                  />
                  {item.title}
                </label>
              ))}
          </fieldset>
          {removable && (
            <Button
              type="button"
              size="small"
              variant="ghost"
              onClick={onRemove}
            >
              Remove {task.title}
            </Button>
          )}
        </div>
      </details>
    </article>
  );
}
