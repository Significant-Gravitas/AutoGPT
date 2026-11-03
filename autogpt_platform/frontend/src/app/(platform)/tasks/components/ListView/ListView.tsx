import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/molecules/Table/components/TablePrimitives";
import { Fragment } from "react";
import { STATUS_LABELS } from "../../data/helpers";
import type { Task } from "../../data/types";
import { groupTasksByStatus } from "./helpers";
import { TaskRow } from "./TaskRow";

interface Props {
  tasks: Task[];
  isLoading: boolean;
  onOpenTask: (id: string) => void;
  emptyLabel: string;
}

const COLUMN_COUNT = 7;

export function ListView({ tasks, isLoading, onOpenTask, emptyLabel }: Props) {
  if (isLoading) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-10 w-full rounded-md" />
        <Skeleton className="h-10 w-full rounded-md" />
        <Skeleton className="h-10 w-full rounded-md" />
      </div>
    );
  }

  if (tasks.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border-base px-4 py-12 text-center">
        <Text variant="body" tone="muted">
          {emptyLabel}
        </Text>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border-base bg-surface">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Key</TableHead>
            <TableHead>Title</TableHead>
            <TableHead>Assignee</TableHead>
            <TableHead>Priority</TableHead>
            <TableHead>Due</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="hidden lg:table-cell">Activity</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {groupTasksByStatus(tasks).map((group) => (
            <Fragment key={group.status}>
              <TableRow>
                <TableCell colSpan={COLUMN_COUNT}>
                  <Text variant="small-medium" tone="emphasis">
                    {STATUS_LABELS[group.status]}{" "}
                    <Text as="span" variant="small" tone="muted">
                      {group.tasks.length}
                    </Text>
                  </Text>
                </TableCell>
              </TableRow>
              {group.tasks.map((task) => (
                <TaskRow key={task.id} task={task} onOpen={onOpenTask} />
              ))}
            </Fragment>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
