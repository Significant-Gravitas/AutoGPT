import type {
  ExpertOption,
  Task,
  TaskActivity,
  TaskArtifact,
  TaskAssignee,
  TaskCounts,
  TaskInbox,
  TaskDetail,
  TaskPlan,
  TaskStatus,
} from "./types";

// Maps the backend payloads (autogpt_platform/TASKS_V1_CONTRACT.md) onto the
// UI types in types.ts.

type Raw = Record<string, unknown>;

function obj(v: unknown): Raw {
  return v && typeof v === "object" ? (v as Raw) : {};
}
function str(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}
function num(v: unknown): number {
  return typeof v === "number" ? v : 0;
}

export function toAssignee(v: unknown): TaskAssignee | null {
  if (!v || typeof v !== "object") return null;
  const r = obj(v);
  const type = (str(r.type) ?? "user") as TaskAssignee["type"] | "system";
  return {
    type: type === "system" ? "otto" : type,
    id: str(r.id),
    name: str(r.name) ?? (type === "otto" ? "Otto" : "Me"),
    avatar_url: str(r.avatar_url),
  };
}

function actor(v: unknown): TaskAssignee {
  return toAssignee(v) ?? { type: "otto", id: null, name: "Otto" };
}

function inboxState(status: TaskStatus): TaskInbox["state"] {
  if (status === "in_progress") return "running";
  if (status === "done") return "done";
  if (status === "cancelled") return "failed";
  return "waiting";
}

function toInbox(v: unknown, status: TaskStatus): TaskInbox {
  const r = obj(v);
  const last = r.last_activity ? obj(r.last_activity) : null;
  return {
    state: (str(r.state) as TaskInbox["state"]) ?? inboxState(status),
    needs_you:
      typeof r.needs_you === "boolean"
        ? r.needs_you
        : status === "blocked" || status === "in_review",
    last_activity: last
      ? {
          kind: str(last.kind) ?? "system",
          text: str(last.text) ?? "",
          actor_name: str(last.actor_name),
          at: str(last.at) ?? new Date().toISOString(),
        }
      : null,
    steps: Array.isArray(r.steps)
      ? r.steps.map((x) => {
          const st = obj(x);
          return {
            label: str(st.label) ?? "",
            time: str(st.time) ?? new Date().toISOString(),
            kind: str(st.kind) ?? "system",
            status: (str(st.status) as TaskInbox["steps"][number]["status"]) ?? null,
          };
        })
      : [],
    tools_used: Array.isArray(r.tools_used)
      ? r.tools_used.map((x) => {
          const t = obj(x);
          return { name: str(t.name) ?? "", title: str(t.title) ?? str(t.name) ?? "", count: num(t.count) };
        })
      : [],
  };
}

export function toTaskCounts(v: unknown, tasks: Task[]): TaskCounts {
  const r = obj(v);
  if (typeof r.all === "number") {
    return {
      all: num(r.all),
      running: num(r.running),
      waiting: num(r.waiting),
      done: num(r.done),
      failed: num(r.failed),
      needs_you: num(r.needs_you),
      by_status: obj(r.by_status) as TaskCounts["by_status"],
    };
  }
  const by_status = {} as TaskCounts["by_status"];
  for (const t of tasks) by_status[t.status] = (by_status[t.status] ?? 0) + 1;
  return {
    all: tasks.length,
    running: tasks.filter((t) => t.inbox.state === "running").length,
    waiting: tasks.filter((t) => t.inbox.state === "waiting").length,
    done: tasks.filter((t) => t.inbox.state === "done").length,
    failed: tasks.filter((t) => t.inbox.state === "failed").length,
    needs_you: tasks.filter((t) => t.inbox.needs_you).length,
    by_status,
  };
}

export function toTask(v: unknown): Task {
  const r = obj(v);
  if ("open_question" in r) {
    const t = r as unknown as Task;
    return {
      ...t,
      inbox: t.inbox ?? toInbox(null, t.status),
      elapsed_ms: t.elapsed_ms ?? null,
      started_at: t.started_at ?? null,
      completed_at: t.completed_at ?? null,
    };
  }
  const metrics = obj(r.metrics);
  const live = obj(r.live);
  const pending = r.pending_input ? obj(r.pending_input) : null;
  const counts = obj(r.subtask_counts);
  return {
    id: String(r.id),
    number: num(r.number),
    title: str(r.title) ?? "",
    description: str(r.description) ?? "",
    status: (str(r.status) ?? "todo") as TaskStatus,
    priority: (str(r.priority) ?? "none") as Task["priority"],
    assignee: toAssignee(r.assignee),
    due_date: str(r.due_date),
    labels: Array.isArray(r.labels) ? (r.labels as string[]) : [],
    parent_id: str(r.parent_id),
    outcome_summary: str(r.outcome_summary),
    budget_usd: typeof r.budget_cap_usd === "number" ? r.budget_cap_usd : null,
    metrics: {
      tool_calls: num(metrics.tool_calls),
      tokens: num(metrics.tokens),
      cost_usd: num(metrics.cost_usd),
      duration_ms: num(metrics.duration_ms),
    },
    live:
      r.live && (live.running || live.current_step)
        ? {
            current_step: str(live.current_step),
            started_at: str(live.turn_started_at) ?? str(r.started_at),
          }
        : null,
    open_question: pending
      ? {
          id: String(pending.id),
          question: str(pending.question) ?? "",
          options: Array.isArray(pending.options)
            ? (pending.options as string[])
            : [],
          asked_by: actor(pending.asked_by),
          asked_at: str(pending.asked_at) ?? new Date().toISOString(),
        }
      : null,
    subtask_counts: { total: num(counts.total), done: num(counts.done) },
    thread_session_id: str(r.work_session_id),
    inbox: toInbox(r.inbox, (str(r.status) ?? "todo") as TaskStatus),
    elapsed_ms: typeof r.elapsed_ms === "number" ? r.elapsed_ms : null,
    started_at: str(r.started_at),
    completed_at: str(r.completed_at),
    created_at: str(r.created_at) ?? new Date().toISOString(),
    updated_at: str(r.updated_at) ?? new Date().toISOString(),
  };
}

function toArtifact(v: unknown): TaskArtifact {
  const r = obj(v);
  const kind = str(r.kind);
  return {
    id: String(r.id),
    kind:
      kind === "url"
        ? "link"
        : kind === "chat" || kind === "run"
          ? kind
          : "file",
    title: str(r.title) ?? "Attachment",
    url:
      str(r.url) ??
      (kind === "chat" && r.ref_id ? `/copilot?sessionId=${r.ref_id}` : null),
  };
}

export function toTaskDetail(v: unknown): TaskDetail {
  const r = obj(v);
  if ("artifacts" in r) return r as unknown as TaskDetail;
  const task = toTask(r);
  const subtasks = Array.isArray(r.subtasks) ? r.subtasks.map(toTask) : [];
  return {
    ...task,
    subtasks: subtasks.map((s) => ({
      id: s.id,
      number: s.number,
      title: s.title,
      status: s.status,
      assignee: s.assignee,
    })),
    artifacts: Array.isArray(r.attachments) ? r.attachments.map(toArtifact) : [],
  };
}

export function toActivity(v: unknown): TaskActivity {
  const r = obj(v);
  const data = obj(r.data);
  const base = {
    id: String(r.id),
    created_at: str(r.created_at) ?? new Date().toISOString(),
    actor: actor(r.actor),
  };
  const body = str(r.body) ?? "";
  switch (r.kind) {
    case "comment":
      return {
        ...base,
        kind: "comment",
        body,
        mentions: Array.isArray(data.mentions)
          ? data.mentions.map((m) => String(obj(m).expert_id ?? obj(m).name))
          : [],
      };
    case "status_changed":
    case "status_change":
      return {
        ...base,
        kind: "status_change",
        from_status: (str(data.from) ?? str(r.from_status) ?? "todo") as TaskStatus,
        to_status: (str(data.to) ?? str(r.to_status) ?? "todo") as TaskStatus,
      };
    case "turn": {
      if ("metrics" in r) return r as unknown as TaskActivity;
      const cards = Array.isArray(data.tool_call_cards) ? data.tool_call_cards : [];
      return {
        ...base,
        kind: "turn",
        summary: str(data.summary),
        metrics: {
          tool_calls: num(data.tool_calls),
          tokens: num(data.tokens),
          cost_usd: num(data.cost_usd),
          duration_ms: num(data.duration_ms),
        },
        tool_calls: cards.map((c) => {
          const card = obj(c);
          const status = str(card.status);
          return {
            id: String(card.id),
            name: str(card.title) ?? str(card.name) ?? "Tool",
            status:
              status === "error" ? "error" : status === "running" ? "running" : "ok",
            input_preview: str(card.input_preview),
            output_preview: str(card.output_preview),
          };
        }),
      };
    }
    case "input_requested":
    case "question":
      return { ...base, kind: "question", body };
    case "input_answered":
    case "answer":
      return { ...base, kind: "answer", body };
    case "created":
      return { ...base, kind: "created", body: body || "Created the task" };
    case "assigned": {
      const to = toAssignee(data.to);
      return {
        ...base,
        kind: "assigned",
        body: body || (to ? `Assigned to ${to.name}` : "Unassigned"),
      };
    }
    case "attachment_added":
    case "artifact_linked": {
      const att = obj(data.attachment);
      return {
        ...base,
        kind: "artifact_linked",
        body: body || `Linked ${str(att.title) ?? "an attachment"}`,
      };
    }
    case "subtask_created":
      return {
        ...base,
        kind: "system",
        body: body || `Added sub-task ${str(data.subtask_key) ?? ""} ${str(data.title) ?? ""}`.trim(),
      };
    case "outcome":
      return { ...base, kind: "system", body: body ? `Outcome: ${body}` : "Finished" };
    case "plan_confirmed":
      return { ...base, kind: "system", body: body || "Plan approved" };
    default:
      return { ...base, kind: "system", body };
  }
}

export function toExpert(v: unknown): ExpertOption {
  const r = obj(v);
  if (typeof r.busy === "boolean") return r as unknown as ExpertOption;
  return {
    id: String(r.id),
    name: str(r.name) ?? "Expert",
    avatar_url: str(r.avatar_url),
    busy: !!r.busy_task_id,
  };
}

/** Accepts the backend `Plan` (goal, steps[].assignee) or a UI TaskPlan. */
export function toPlan(v: unknown): TaskPlan | null {
  const r = obj(v);
  if (!r.id || !Array.isArray(r.steps)) return null;
  const status = str(r.status);
  return {
    id: String(r.id),
    title: str(r.title) ?? str(r.goal) ?? "Plan",
    summary: str(r.summary) ?? undefined,
    steps: r.steps.map((s) => {
      const step = obj(s);
      return {
        title: str(step.title) ?? "",
        description: str(step.description) ?? undefined,
        expert: toAssignee(step.expert ?? step.assignee),
      };
    }),
    budget_usd: typeof r.budget_usd === "number" ? r.budget_usd : null,
    status:
      status === "confirmed" || status === "approved"
        ? "approved"
        : status === "cancelled" || status === "rejected"
          ? "rejected"
          : "proposed",
    task_id: str(r.parent_task_id) ?? str(r.task_id),
  };
}
