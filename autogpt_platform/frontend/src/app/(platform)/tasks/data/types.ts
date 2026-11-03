export type TaskStatus =
  | "backlog"
  | "todo"
  | "in_progress"
  | "blocked"
  | "in_review"
  | "done"
  | "cancelled";

export type TaskPriority = "none" | "low" | "medium" | "high" | "urgent";

export type AssigneeType = "user" | "expert" | "otto";

export interface TaskAssignee {
  type: AssigneeType;
  id: string | null;
  name: string;
  avatar_url?: string | null;
}

export interface TaskMetrics {
  tool_calls: number;
  tokens: number;
  cost_usd: number;
  duration_ms: number;
}

export interface TaskLiveStrip {
  current_step: string | null;
  started_at: string | null;
}

export interface TaskQuestion {
  id: string;
  question: string;
  options?: string[];
  asked_by: TaskAssignee;
  asked_at: string;
}

export interface TaskArtifact {
  id: string;
  kind: "file" | "chat" | "run" | "link";
  title: string;
  url?: string | null;
}

export interface TaskSubtaskSummary {
  id: string;
  number: number;
  title: string;
  status: TaskStatus;
  assignee: TaskAssignee | null;
}

export interface Task {
  id: string;
  number: number;
  title: string;
  description: string;
  status: TaskStatus;
  priority: TaskPriority;
  assignee: TaskAssignee | null;
  due_date: string | null;
  labels: string[];
  parent_id: string | null;
  outcome_summary: string | null;
  budget_usd: number | null;
  metrics: TaskMetrics;
  live: TaskLiveStrip | null;
  open_question: TaskQuestion | null;
  subtask_counts: { total: number; done: number };
  thread_session_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface TaskDetail extends Task {
  subtasks: TaskSubtaskSummary[];
  artifacts: TaskArtifact[];
}

export interface ToolCallSummary {
  id: string;
  name: string;
  status: "ok" | "error" | "running";
  input_preview?: string | null;
  output_preview?: string | null;
}

export type TaskActivity =
  | {
      id: string;
      kind: "comment";
      created_at: string;
      actor: TaskAssignee;
      body: string;
      mentions: string[];
    }
  | {
      id: string;
      kind: "status_change";
      created_at: string;
      actor: TaskAssignee;
      from_status: TaskStatus;
      to_status: TaskStatus;
    }
  | {
      id: string;
      kind: "turn";
      created_at: string;
      actor: TaskAssignee;
      summary: string | null;
      metrics: TaskMetrics;
      tool_calls: ToolCallSummary[];
    }
  | {
      id: string;
      kind: "question" | "answer";
      created_at: string;
      actor: TaskAssignee;
      body: string;
    }
  | {
      id: string;
      kind: "created" | "assigned" | "artifact_linked" | "system";
      created_at: string;
      actor: TaskAssignee;
      body: string;
    };

export interface TaskFilters {
  status?: TaskStatus[];
  priority?: TaskPriority[];
  assignee_ids?: string[];
  due?: "overdue" | "today" | "week" | null;
  mine?: boolean;
  parent_id?: string | null;
}

export interface CreateTaskInput {
  title: string;
  description?: string;
  assignee?: { type: AssigneeType; id: string | null } | null;
  priority?: TaskPriority;
  due_date?: string | null;
  labels?: string[];
  parent_id?: string | null;
  budget_usd?: number | null;
}

export type UpdateTaskInput = Partial<
  Omit<CreateTaskInput, "parent_id"> & { status: TaskStatus }
>;

export interface ExpertOption {
  id: string;
  name: string;
  avatar_url?: string | null;
  busy: boolean;
}

export interface TaskPlanStep {
  title: string;
  description?: string;
  expert: TaskAssignee | null;
}

export interface TaskPlan {
  id: string;
  title: string;
  summary?: string;
  steps: TaskPlanStep[];
  budget_usd: number | null;
  status: "proposed" | "approved" | "rejected";
  task_id?: string | null;
}

/** Event names pushed on the notification WebSocket. */
export const TASK_EVENTS = [
  "task_created",
  "task_updated",
  "task_deleted",
  "task_activity",
  "task_needs_input",
  "task_completed",
] as const;
export type TaskEventName = (typeof TASK_EVENTS)[number];
