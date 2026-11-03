import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Wrench01Icon } from "@hugeicons/core-free-icons";
import { motion } from "framer-motion";
import { taskKey } from "../../data/helpers";
import type { Task } from "../../data/types";
import { DueChip } from "../DueChip/DueChip";
import { PriorityGlyph } from "../PriorityGlyph/PriorityGlyph";
import { SubtaskRing } from "../SubtaskRing/SubtaskRing";
import { TaskAssigneeAvatar } from "../TaskAssigneeAvatar/TaskAssigneeAvatar";
import { LiveClock } from "./LiveClock";

interface Props {
  task: Task;
  onOpen: (id: string) => void;
}

export function TaskCard({ task, onOpen }: Props) {
  const isRunning = task.status === "in_progress";
  const needsYou = task.status === "blocked";
  const isDone = task.status === "done";

  return (
    <motion.button
      layout
      layoutId={`task-card-${task.id}`}
      transition={{ type: "spring", stiffness: 500, damping: 40 }}
      type="button"
      onClick={() => onOpen(task.id)}
      className={cn(
        "flex w-full flex-col gap-2 rounded-lg border bg-surface p-3 text-left shadow-sm transition-colors",
        "hover:border-border-strong focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none",
        needsYou ? "border-warning" : "border-border-base",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <Text variant="small-mono" tone="muted">
          {taskKey(task.number)}
        </Text>
        <PriorityGlyph priority={task.priority} />
      </div>
      <Text variant="body" weight="semibold" className="line-clamp-2">
        {task.title}
      </Text>
      {needsYou && task.open_question ? (
        <Text variant="small" tone="warning" className="line-clamp-2">
          {task.open_question.question}
        </Text>
      ) : task.description ? (
        <Text variant="small" tone="secondary" className="line-clamp-2">
          {task.description}
        </Text>
      ) : null}
      <div className="flex items-center gap-2">
        <TaskAssigneeAvatar assignee={task.assignee} />
        {task.due_date ? <DueChip due={task.due_date} isDone={isDone} /> : null}
        <span className="flex-1" />
        {isRunning && task.live?.started_at ? (
          <LiveClock startedAt={task.live.started_at} />
        ) : null}
        {task.metrics.tool_calls > 0 ? (
          <span
            className="inline-flex items-center gap-1"
            title={`${task.metrics.tool_calls} tool calls`}
          >
            <Icon icon={Wrench01Icon} size={12} tone="muted" />
            <Text variant="small" tone="secondary">
              {task.metrics.tool_calls}
            </Text>
          </span>
        ) : null}
        <SubtaskRing
          done={task.subtask_counts.done}
          total={task.subtask_counts.total}
        />
      </div>
    </motion.button>
  );
}
