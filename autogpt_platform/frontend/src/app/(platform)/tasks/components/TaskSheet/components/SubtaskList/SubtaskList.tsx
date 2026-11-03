"use client";

import { Add01Icon } from "@hugeicons/core-free-icons";
import type { FormEvent, KeyboardEvent } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { isKey } from "@/lib/keyboard";
import { STATUS_LABELS, taskKey } from "../../../../data/helpers";
import type { TaskDetail } from "../../../../data/types";
import { STATUS_ICONS } from "../../helpers";
import { AssigneeAvatar } from "../AssigneeAvatar/AssigneeAvatar";
import { useSubtaskList } from "./useSubtaskList";

interface Props {
  task: TaskDetail;
  onOpenTask?: (taskId: string) => void;
}

export function SubtaskList({ task, onOpenTask }: Props) {
  const {
    isAdding,
    startAdding,
    cancel,
    title,
    setTitle,
    submit,
    isCreating,
    isError,
  } = useSubtaskList(task.id);
  const canAdd = !task.parent_id;
  const subtasks = task.subtasks;
  const doneCount = subtasks.filter((s) => s.status === "done").length;

  if (!subtasks.length && !canAdd) return null;

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (isKey(event, "Escape")) {
      event.stopPropagation();
      cancel();
    }
  }

  return (
    <section aria-label="Sub-tasks" className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <Text variant="small-semibold" tone="secondary">
          Sub-tasks
        </Text>
        {subtasks.length ? (
          <Text variant="small" tone="muted" tabular>
            {doneCount} of {subtasks.length} done
          </Text>
        ) : null}
      </div>
      {subtasks.length ? (
        <ul className="flex flex-col overflow-hidden rounded-lg border border-border-base">
          {subtasks.map((subtask) => {
            const status = STATUS_ICONS[subtask.status];
            return (
              <li
                key={subtask.id}
                className="border-b border-border-base last:border-b-0"
              >
                <button
                  type="button"
                  onClick={() => onOpenTask?.(subtask.id)}
                  className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-surface-hover focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none focus-visible:ring-inset"
                >
                  <Icon
                    icon={status.icon}
                    tone={status.tone}
                    aria-label={STATUS_LABELS[subtask.status]}
                  />
                  <Text variant="small-mono" tone="muted" as="span">
                    {taskKey(subtask.number)}
                  </Text>
                  <Text
                    variant="body-tight"
                    as="span"
                    tone={subtask.status === "done" ? "muted" : "primary"}
                    className="min-w-0 flex-1 truncate"
                  >
                    {subtask.title}
                  </Text>
                  {subtask.assignee ? (
                    <AssigneeAvatar assignee={subtask.assignee} size={18} />
                  ) : null}
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
      {canAdd ? (
        isAdding ? (
          <form onSubmit={handleSubmit} className="flex items-center gap-2">
            <Input
              bare
              autoFocus
              aria-label="Sub-task title"
              placeholder="Sub-task title"
              className="flex-1"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isCreating}
            />
            <Button
              type="submit"
              variant="primary"
              size="row"
              loading={isCreating}
              disabled={!title.trim()}
            >
              Add
            </Button>
            <Button
              type="button"
              variant="ghost-muted"
              size="row"
              onClick={cancel}
            >
              Cancel
            </Button>
          </form>
        ) : (
          <Button
            variant="ghost-muted"
            size="xs"
            leadingIcon={Add01Icon}
            onClick={startAdding}
            className="self-start"
          >
            Add sub-task
          </Button>
        )
      ) : null}
      {isError ? (
        <Text variant="small" tone="error">
          The sub-task could not be added. Try again.
        </Text>
      ) : null}
    </section>
  );
}
