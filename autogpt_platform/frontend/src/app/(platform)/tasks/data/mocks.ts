import type {
  CreateTaskInput,
  ExpertOption,
  Task,
  TaskActivity,
  TaskAssignee,
  TaskDetail,
  TaskFilters,
  TaskPlan,
  TaskState,
  TaskStatus,
  UpdateTaskInput,
} from "./types";

// In-memory stand-in for the tasks API, used only while the backend routes
// are missing (the client falls back here on a 404 or network error).

const ME: TaskAssignee = { type: "user", id: "me", name: "You" };
const OTTO: TaskAssignee = { type: "otto", id: null, name: "Otto" };

export const MOCK_EXPERTS: ExpertOption[] = [
  { id: "exp-research", name: "Riley (Research)", busy: true },
  { id: "exp-writer", name: "Wren (Writing)", busy: false },
  { id: "exp-sales", name: "Sam (Sales ops)", busy: true },
  { id: "exp-data", name: "Dana (Data)", busy: false },
];

function expert(id: string): TaskAssignee {
  const e = MOCK_EXPERTS.find((x) => x.id === id)!;
  return { type: "expert", id: e.id, name: e.name };
}

const now = Date.now();
const iso = (offsetMin: number) =>
  new Date(now + offsetMin * 60_000).toISOString();

const MOCK_STATE: Record<TaskStatus, TaskState> = {
  backlog: "working",
  todo: "working",
  in_progress: "working",
  blocked: "needs_you",
  in_review: "needs_you",
  done: "done",
  cancelled: "stopped",
};

function base(
  n: number,
  partial: Partial<Task> & Pick<Task, "title" | "status">,
): Task {
  return {
    id: `task-${n}`,
    number: n,
    description: "",
    priority: "medium",
    assignee: null,
    due_date: null,
    labels: [],
    parent_id: null,
    outcome_summary: null,
    budget_usd: 2,
    metrics: { tool_calls: 0, tokens: 0, cost_usd: 0, duration_ms: 0 },
    live: null,
    open_question: null,
    subtask_counts: { total: 0, done: 0 },
    thread_session_id: null,
    plan_id: null,
    inbox: {
      state: "waiting",
      needs_you: false,
      last_activity: null,
      steps: [],
      tools_used: [],
    },
    state: MOCK_STATE[partial.status],
    raw_status: "",
    now: "",
    ask: null,
    blocker: null,
    approvals: [],
    elapsed_ms: null,
    started_at: null,
    completed_at: null,
    created_at: iso(-600 + n),
    updated_at: iso(-60 + n),
    ...partial,
  };
}

let tasks: Task[] = [
  base(101, {
    title: "Research top 10 competitors' pricing pages",
    description:
      "Collect pricing tiers, seat limits and annual discounts for the ten closest competitors. Put it in a comparison table.",
    status: "in_progress",
    priority: "high",
    assignee: expert("exp-research"),
    due_date: iso(60 * 24),
    labels: ["research"],
    metrics: {
      tool_calls: 14,
      tokens: 48_200,
      cost_usd: 0.31,
      duration_ms: 412_000,
    },
    live: { current_step: "Reading pricing page 7 of 10", started_at: iso(-7) },
    subtask_counts: { total: 3, done: 1 },
  }),
  base(102, {
    title: "Draft launch announcement for the new export feature",
    description: "Blog post plus a short LinkedIn version, friendly tone.",
    status: "blocked",
    priority: "medium",
    assignee: expert("exp-writer"),
    due_date: iso(60 * 6),
    metrics: {
      tool_calls: 4,
      tokens: 12_100,
      cost_usd: 0.04,
      duration_ms: 80_000,
    },
    open_question: {
      id: "q-1",
      question:
        "Should the post mention the CSV limit of 50k rows, or keep it vague?",
      options: ["Mention the limit", "Keep it vague"],
      asked_by: expert("exp-writer"),
      asked_at: iso(-12),
    },
  }),
  base(103, {
    title: "Clean up duplicate leads in the CRM",
    description: "Merge duplicates by email domain and flag conflicts.",
    status: "todo",
    priority: "low",
    assignee: expert("exp-sales"),
    labels: ["crm"],
  }),
  base(104, {
    title: "Weekly metrics digest",
    description: "Summarise signups, activation and churn for last week.",
    status: "done",
    priority: "medium",
    assignee: expert("exp-data"),
    outcome_summary:
      "Signups up 12% week over week; activation flat at 41%; churn down to 2.1%. Digest posted to #metrics.",
    metrics: {
      tool_calls: 9,
      tokens: 30_400,
      cost_usd: 0.12,
      duration_ms: 205_000,
    },
  }),
  base(105, {
    title: "Decide on Q4 webinar topic",
    description: "",
    status: "todo",
    priority: "urgent",
    assignee: ME,
    due_date: iso(-60 * 5),
  }),
  base(106, {
    title: "Plan the onboarding email sequence",
    status: "backlog",
    priority: "none",
    assignee: OTTO,
  }),
  base(107, {
    title: "Pull pricing for Competitor A",
    status: "done",
    priority: "medium",
    assignee: expert("exp-research"),
    parent_id: "task-101",
  }),
  base(108, {
    title: "Pull pricing for Competitor B",
    status: "in_progress",
    priority: "medium",
    assignee: expert("exp-research"),
    parent_id: "task-101",
  }),
  base(109, {
    title: "Build the comparison table",
    status: "todo",
    priority: "medium",
    assignee: expert("exp-research"),
    parent_id: "task-101",
  }),
];

const activity: Record<string, TaskActivity[]> = {
  "task-101": [
    {
      id: "a1",
      kind: "created",
      created_at: iso(-30),
      actor: ME,
      body: "Created the task",
    },
    {
      id: "a2",
      kind: "status_change",
      created_at: iso(-29),
      actor: expert("exp-research"),
      from_status: "todo",
      to_status: "in_progress",
    },
    {
      id: "a3",
      kind: "turn",
      created_at: iso(-20),
      actor: expert("exp-research"),
      summary: "Found pricing pages for 6 competitors",
      metrics: {
        tool_calls: 4,
        tokens: 12_000,
        cost_usd: 0.04,
        duration_ms: 80_000,
      },
      tool_calls: [
        {
          id: "t1",
          name: "web_search",
          status: "ok",
          input_preview: "competitor pricing SaaS",
          output_preview: "10 results",
        },
        {
          id: "t2",
          name: "web_fetch",
          status: "ok",
          input_preview: "https://example.com/pricing",
          output_preview: "3 tiers found",
        },
        {
          id: "t3",
          name: "web_fetch",
          status: "error",
          input_preview: "https://example.org/pricing",
          output_preview: "403 Forbidden",
        },
        {
          id: "t4",
          name: "write_workspace_file",
          status: "ok",
          input_preview: "pricing.csv",
          output_preview: "saved",
        },
      ],
    },
    {
      id: "a4",
      kind: "comment",
      created_at: iso(-10),
      actor: ME,
      body: "@Riley (Research) include annual discounts too",
      mentions: ["exp-research"],
    },
  ],
  "task-102": [
    {
      id: "b1",
      kind: "question",
      created_at: iso(-12),
      actor: expert("exp-writer"),
      body: "Should the post mention the CSV limit of 50k rows, or keep it vague?",
    },
  ],
};

let nextNumber = 110;

function detail(t: Task): TaskDetail {
  const subs = tasks.filter((s) => s.parent_id === t.id);
  return {
    ...t,
    subtask_counts: subs.length
      ? {
          total: subs.length,
          done: subs.filter((s) => s.status === "done").length,
        }
      : t.subtask_counts,
    subtasks: subs.map((s) => ({
      id: s.id,
      number: s.number,
      title: s.title,
      status: s.status,
      assignee: s.assignee,
    })),
    artifacts:
      t.status === "done"
        ? [{ id: "f1", kind: "file", title: "metrics-digest.md", url: null }]
        : [],
  };
}

function assigneeFrom(input: CreateTaskInput["assignee"]): TaskAssignee | null {
  if (!input) return null;
  if (input.type === "user") return ME;
  if (input.type === "otto") return OTTO;
  return input.id ? expert(input.id) : null;
}

export const mockApi = {
  list(filters: TaskFilters): Task[] {
    return tasks.filter((t) => {
      if (filters.parent_id === undefined && t.parent_id) return false;
      if (filters.status?.length && !filters.status.includes(t.status))
        return false;
      if (filters.priority?.length && !filters.priority.includes(t.priority))
        return false;
      if (
        filters.assignee_ids?.length &&
        !filters.assignee_ids.includes(t.assignee?.id ?? "")
      )
        return false;
      if (filters.mine && t.assignee?.type !== "user") return false;
      return true;
    });
  },
  get(id: string): TaskDetail {
    const t = tasks.find((x) => x.id === id);
    if (!t) throw new Error("Task not found");
    return detail(t);
  },
  activity(id: string): TaskActivity[] {
    return activity[id] ?? [];
  },
  create(input: CreateTaskInput): Task {
    const assignee = assigneeFrom(input.assignee);
    const t = base(nextNumber++, {
      title: input.title,
      description: input.description ?? "",
      priority: input.priority ?? "none",
      due_date: input.due_date ?? null,
      labels: input.labels ?? [],
      parent_id: input.parent_id ?? null,
      assignee,
      status: assignee?.type === "expert" ? "in_progress" : "todo",
      live:
        assignee?.type === "expert"
          ? {
              current_step: "Getting started",
              started_at: new Date().toISOString(),
            }
          : null,
    });
    tasks = [t, ...tasks];
    activity[t.id] = [
      {
        id: `c-${t.id}`,
        kind: "created",
        created_at: t.created_at,
        actor: ME,
        body: "Created the task",
      },
    ];
    return t;
  },
  update(id: string, input: UpdateTaskInput): Task {
    tasks = tasks.map((t) =>
      t.id === id
        ? ({
            ...t,
            ...input,
            assignee:
              input.assignee !== undefined
                ? assigneeFrom(input.assignee)
                : t.assignee,
            updated_at: new Date().toISOString(),
          } as Task)
        : t,
    );
    return tasks.find((t) => t.id === id)!;
  },
  comment(id: string, body: string): TaskActivity {
    const mentions = MOCK_EXPERTS.filter((e) =>
      body.includes(`@${e.name}`),
    ).map((e) => e.id);
    const a: TaskActivity = {
      id: `cm-${Date.now()}`,
      kind: "comment",
      created_at: new Date().toISOString(),
      actor: ME,
      body,
      mentions,
    };
    activity[id] = [...(activity[id] ?? []), a];
    return a;
  },
  answer(id: string, answer: string): Task {
    activity[id] = [
      ...(activity[id] ?? []),
      {
        id: `ans-${Date.now()}`,
        kind: "answer",
        created_at: new Date().toISOString(),
        actor: ME,
        body: answer,
      },
    ];
    tasks = tasks.map((t) =>
      t.id === id ? { ...t, open_question: null, status: "in_progress" } : t,
    );
    return tasks.find((t) => t.id === id)!;
  },
  experts(): ExpertOption[] {
    return MOCK_EXPERTS;
  },
  approvePlan(plan: TaskPlan): Task {
    const parent = mockApi.create({
      title: plan.title,
      description: plan.summary,
      assignee: { type: "otto", id: null },
    });
    plan.steps.forEach((s) =>
      mockApi.create({
        title: s.title,
        description: s.description,
        parent_id: parent.id,
        assignee: s.expert ? { type: s.expert.type, id: s.expert.id } : null,
      }),
    );
    return parent;
  },
};
