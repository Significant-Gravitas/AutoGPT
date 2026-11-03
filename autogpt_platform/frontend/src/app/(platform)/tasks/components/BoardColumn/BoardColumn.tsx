import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { AnimatePresence } from "framer-motion";
import type { Task } from "../../data/types";
import { TaskCard } from "../TaskCard/TaskCard";

const COLUMN_DOTS: Record<string, string> = {
  in_progress: "bg-info",
  needs_you: "bg-warning",
  done: "bg-success",
};

interface Props {
  id: string;
  label: string;
  tasks: Task[];
  isLoading: boolean;
  onOpenTask: (id: string) => void;
}

export function BoardColumn({
  id,
  label,
  tasks,
  isLoading,
  onOpenTask,
}: Props) {
  return (
    <section
      aria-label={label}
      className="flex w-72 shrink-0 flex-col gap-2 rounded-xl bg-surface-muted p-2 md:w-auto md:min-w-0 md:flex-1"
    >
      <header className="flex items-center gap-2 px-1 py-1">
        <span
          aria-hidden="true"
          className={cn(
            "size-2 rounded-full",
            COLUMN_DOTS[id] ?? "bg-foreground-subtle",
          )}
        />
        <Text variant="small-medium" tone="emphasis">
          {label}
        </Text>
        <Text variant="small" tone="muted">
          {isLoading ? "" : tasks.length}
        </Text>
      </header>
      <div className="flex flex-col gap-2">
        {isLoading ? (
          <>
            <Skeleton className="h-24 w-full rounded-lg" />
            <Skeleton className="h-24 w-full rounded-lg" />
          </>
        ) : tasks.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border-base px-3 py-6 text-center">
            <Text variant="small" tone="muted">
              No tasks
            </Text>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {tasks.map((task) => (
              <TaskCard key={task.id} task={task} onOpen={onOpenTask} />
            ))}
          </AnimatePresence>
        )}
      </div>
    </section>
  );
}
