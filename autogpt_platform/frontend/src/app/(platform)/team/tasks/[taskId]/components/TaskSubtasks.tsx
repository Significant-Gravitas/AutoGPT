"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import Link from "next/link";
import { TaskStatusChip } from "../../../components/TaskStatusChip/TaskStatusChip";
import {
  buildTaskTree,
  formatSpend,
  getTaskOrbVariant,
} from "../../../task-helpers";

interface Props {
  taskId: string;
  childTasks: DelegatedTask[];
}

export function TaskSubtasks({ taskId, childTasks }: Props) {
  if (childTasks.length === 0) return null;

  const done = childTasks.filter((task) => task.status === "DONE").length;

  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <Text variant="small" className="font-medium text-zinc-900">
          Subtasks
        </Text>
        <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] tabular-nums text-zinc-500">
          {done}/{childTasks.length}
        </span>
      </div>

      <ul
        className="overflow-hidden rounded-xl ring-[0.5px] ring-zinc-200"
        aria-label="Subtasks"
      >
        {buildTaskTree(taskId, childTasks).map(({ task: child, depth }) => (
          <li
            key={child.id}
            className="border-b-[0.5px] border-zinc-200 last:border-b-0"
          >
            <Link
              href={`/team/tasks/${child.id}`}
              style={{ paddingLeft: 12 + (depth - 1) * 20 }}
              className="flex items-center gap-2.5 bg-white py-2.5 pr-3 transition-colors hover:bg-zinc-50"
            >
              <ExpertAvatar
                name={child.owner?.name ?? null}
                avatarUrl={child.owner?.avatar_url ?? null}
                size={20}
              />
              <span className="min-w-0 flex-1 truncate text-[13px] text-zinc-900">
                {child.title}
              </span>
              <span className="shrink-0 text-[13px] tabular-nums text-zinc-400">
                {formatSpend(child.spend_total)}
              </span>
              <TaskStatusChip
                status={child.status}
                orbVariant={getTaskOrbVariant(child.id)}
              />
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
