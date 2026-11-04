"use client";

import { Add01Icon } from "@hugeicons/core-free-icons";
import type { FormEvent, KeyboardEvent } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { isKey } from "@/lib/keyboard";
import { STATUS_LABELS, taskKey } from "../../../../data/helpers";
import type { TaskDetail } from "../../../../data/types";
import { STATUS_ICONS } from "../../helpers";
import { PanelSection } from "../PanelSection/PanelSection";
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
    <PanelSection
      label="Sub-tasks"
      meta={
        subtasks.length ? `${doneCount} of ${subtasks.length} done` : undefined
      }
    >
      {subtasks.length ? (
        <ul className="flex flex-col divide-y divide-border-hairline overflow-hidden rounded-card bg-surface shadow-card">
          {subtasks.map((subtask) => {
            const status = STATUS_ICONS[subtask.status];
            return (
              <li key={subtask.id}>
                <button
                  type="button"
                  onClick={() => onOpenTask?.(subtask.id)}
                  className="flex min-h-9 w-full items-center gap-2.5 px-3 text-left transition-[background-color] duration-150 ease-out-expo hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset"
                >
                  <Icon
                    icon={status.icon}
                    tone={status.tone}
                    size={14}
                    aria-label={STATUS_LABELS[subtask.status]}
                  />
                  <Text variant="small-mono" tone="faint" as="span">
                    {taskKey(subtask.number)}
                  </Text>
                  <Text
                    variant="inbox"
                    as="span"
                    tone={subtask.status === "done" ? "quiet" : "ink"}
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
            <input
              type="text"
              autoFocus
              aria-label="Sub-task title"
              placeholder="Sub-task title"
              className="h-8 w-full min-w-0 flex-1 rounded-control bg-surface px-2.5 text-body-sm text-foreground-ink shadow-card transition-shadow duration-150 ease-out-expo placeholder:text-foreground-faint focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50"
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
    </PanelSection>
  );
}
