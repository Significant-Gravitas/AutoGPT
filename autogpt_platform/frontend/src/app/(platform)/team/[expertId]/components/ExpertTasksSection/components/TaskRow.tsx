"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  formatElapsed,
  formatSpend,
  getStatusLabel,
  getStatusVariant,
} from "../../../../task-helpers";

interface Props {
  task: DelegatedTask;
}

export function TaskRow({ task }: Props) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-2xl bg-white p-3 ring-1 ring-inset ring-zinc-200">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-zinc-900">
          {task.title}
        </p>
        <div className="mt-1 flex items-center gap-2">
          <Badge variant={getStatusVariant(task.status)} size="small">
            {getStatusLabel(task.status)}
          </Badge>
          <Text variant="small" className="text-zinc-500">
            {formatElapsed(task)}
          </Text>
          <Text variant="small" className="text-zinc-500">
            {formatSpend(task.spend_total)}
          </Text>
        </div>
      </div>
      <Button
        as="NextLink"
        href={`/team/tasks/${task.id}`}
        variant="secondary"
        size="small"
      >
        Open
      </Button>
    </div>
  );
}
