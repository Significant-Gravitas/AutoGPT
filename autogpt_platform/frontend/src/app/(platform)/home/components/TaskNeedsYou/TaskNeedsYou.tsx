"use client";

import { Task01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import type { Task } from "@/app/api/__generated__/models/task";
import { taskHref } from "@/app/(platform)/copilot/tools/CreateTask/helpers";
import type { TaskOwner } from "@/app/(platform)/tasks/helpers";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatTimeAgo } from "@/lib/utils/time";
import { HomeTile } from "../HomeTile/HomeTile";
import { useTaskNeedsYou } from "./useTaskNeedsYou";

interface Props {
  className?: string;
}

/** Tasks waiting on the user; nothing at all when none are. */
export function TaskNeedsYou({ className }: Props) {
  const { rows } = useTaskNeedsYou();

  if (rows.length === 0) return null;

  return (
    <HomeTile
      className={className}
      icon={Task01Icon}
      title="Tasks that need you"
      badge={
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className="size-1.5 shrink-0 rounded-full bg-status-waiting"
          />
          <Text variant="inbox-medium" as="span" tone="waiting" tabular>
            {rows.length}
          </Text>
        </span>
      }
    >
      <ul className="divide-y divide-border-hairline">
        {rows.map(({ task, owner }) => (
          <li key={task.id}>
            <TaskRow task={task} owner={owner} />
          </li>
        ))}
      </ul>
    </HomeTile>
  );
}

interface RowProps {
  task: Task;
  owner: TaskOwner;
}

function TaskRow({ task, owner }: RowProps) {
  return (
    <NextLink
      href={taskHref(task.id)}
      className="flex items-start gap-3 px-4 py-3 transition-[background-color] duration-150 ease-out-expo hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset"
    >
      <span className="mt-0.5 shrink-0">
        <ExpertAvatar name={owner.name} avatarUrl={owner.avatarUrl} size={24} />
      </span>
      <div className="min-w-0 flex-1">
        <Text variant="inbox-medium" tone="ink" className="truncate">
          {task.title}
        </Text>
        <Text variant="small" tone="quiet">
          {owner.name} · {formatTimeAgo(String(task.updated_at))}
        </Text>
      </div>
    </NextLink>
  );
}
