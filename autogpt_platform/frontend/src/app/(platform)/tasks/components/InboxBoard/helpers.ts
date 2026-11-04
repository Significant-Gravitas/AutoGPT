import type { AgentTaskStatus } from "@/components/atoms/agent-inbox-clean";
import { shortAgo } from "../../data/helpers";
import type { Task, TaskState } from "../../data/types";
import { assigneeLabel } from "../TaskAssigneeAvatar/TaskAssigneeAvatar";

const STATUS: Record<TaskState, AgentTaskStatus> = {
  working: "running",
  needs_you: "waiting",
  stuck: "waiting",
  done: "done",
  stopped: "cancelled",
};

/** What the inbox card shows for a task, without the React parts. */
export function toInboxFields(task: Task) {
  return {
    id: task.id,
    title: task.title,
    description: task.now || undefined,
    agent: task.assignee ? assigneeLabel(task.assignee) : "Otto",
    status: STATUS[task.state],
    startedAt: Date.parse(task.started_at ?? task.created_at),
    elapsed: task.elapsed_ms ?? undefined,
    tools: task.metrics.tool_calls || undefined,
    steps: (task.inbox?.steps ?? []).map((step) => ({
      label: step.label,
      time: step.time ? shortAgo(step.time) : undefined,
    })),
  };
}

/** A waiting card says what it waits on in our own words and buttons. */
export function needsExtra(task: Task) {
  return (
    task.approvals.length > 0 ||
    !!task.ask ||
    !!task.blocker ||
    task.raw_status === "awaiting_accept"
  );
}
