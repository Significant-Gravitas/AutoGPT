"use client";

import { Add01Icon, Task01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { taskKey } from "@/app/(platform)/tasks/data/helpers";
import type { Task } from "@/app/(platform)/tasks/data/types";
import { AssigneeAvatar } from "@/app/(platform)/copilot/tools/TaskPlan/components/AssigneeAvatar";
import { HomeTile } from "../HomeTile/HomeTile";
import { useTaskNeedsYou } from "./useTaskNeedsYou";

interface Props {
  className?: string;
}

export function TaskNeedsYou({ className }: Props) {
  const { tasks, isLoading } = useTaskNeedsYou();

  return (
    <HomeTile
      className={className}
      icon={Task01Icon}
      title="Tasks that need you"
      badge={
        tasks.length > 0 ? (
          <span className="rounded-md bg-warning-surface px-1.5 py-0.5">
            <Text
              variant="small-medium"
              as="span"
              tone="warning"
              tabular
              className="block"
            >
              {tasks.length}
            </Text>
          </span>
        ) : null
      }
      meta={
        <Button
          as="NextLink"
          href="/tasks?new=1"
          variant="ghost-muted"
          size="xs"
          leftIcon={<Icon icon={Add01Icon} size={14} />}
        >
          Create task
        </Button>
      }
    >
      {tasks.length === 0 ? (
        <div className="px-4 py-3">
          <Text variant="small" tone="muted">
            {isLoading ? "Loading…" : "Nothing needs you right now."}
          </Text>
        </div>
      ) : (
        <ul className="divide-y divide-separator">
          {tasks.map((task) => (
            <li key={task.id}>
              <TaskRow task={task} />
            </li>
          ))}
        </ul>
      )}
    </HomeTile>
  );
}

function TaskRow({ task }: { task: Task }) {
  const asker = task.open_question?.asked_by ?? task.assignee;
  return (
    <NextLink
      href={`/tasks?task=${encodeURIComponent(task.id)}`}
      className="flex items-start gap-3 px-4 py-3 transition-colors hover:bg-surface-hover focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none focus-visible:ring-inset"
    >
      {asker ? (
        <span className="mt-0.5 shrink-0">
          <AssigneeAvatar assignee={asker} size={24} />
        </span>
      ) : null}
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-2">
          <Text variant="small-medium" as="span" tone="muted" tabular>
            {taskKey(task.number)}
          </Text>
          <Text
            variant="body-medium"
            as="span"
            tone="primary"
            className="truncate"
          >
            {task.title}
          </Text>
        </div>
        {task.open_question ? (
          <Text variant="small" tone="body" className="line-clamp-2">
            {task.open_question.question}
          </Text>
        ) : null}
        {asker ? (
          <Text variant="small" tone="muted">
            {asker.name}
          </Text>
        ) : null}
      </div>
    </NextLink>
  );
}
