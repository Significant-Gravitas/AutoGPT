import type { CardDecision } from "@/app/api/__generated__/models/cardDecision";
import type { EvidenceDetail } from "@/app/api/__generated__/models/evidenceDetail";
import type { TaskActionRequest } from "@/app/api/__generated__/models/taskActionRequest";
import type { TaskSummary } from "@/app/api/__generated__/models/taskSummary";
import { customMutator } from "@/app/api/mutators/custom-mutator";
import { matchesAssignees, OTTO_FILTER_ID } from "./helpers";
import { mockApi } from "./mocks";
import {
  summaryFromTask,
  toActivity,
  toExpert,
  toPlan,
  toTask,
  toTaskCounts,
  toTaskDetail,
} from "./normalize";

import type {
  CreateTaskInput,
  TaskFilters,
  TaskPlan,
  UpdateTaskInput,
} from "./types";

const ACTIVITY_PAGE = 50;

// Thin client for the Tasks v1 routes (see autogpt_platform/TASKS_V1_CONTRACT.md).
// Until the backend serves them, a 404 or network failure falls back to the
// in-memory mocks so the UI stays usable.

type Envelope<T> = { data: T; status: number; headers: Headers };

let useMocks = process.env.NEXT_PUBLIC_TASKS_V1_MOCKS === "true";

function isMissingRoute(error: unknown) {
  if (!error || typeof error !== "object") return true;
  const status = (error as { status?: number }).status;
  return status === undefined || status === 404 || status === 405;
}

async function call<T>(
  url: string,
  init: RequestInit,
  fallback: () => T,
): Promise<T> {
  if (useMocks) return fallback();
  try {
    const res = await customMutator<Envelope<T>>(url, init);
    return res.data;
  } catch (error) {
    if (isMissingRoute(error)) {
      useMocks = true;
      return fallback();
    }
    throw error;
  }
}

/** Like `call`, for writes that must reach the server: a missing route is a
 *  failure the caller rolls back, not a reason to switch to the mocks. */
async function send<T>(url: string, init: RequestInit, mock: () => T) {
  if (useMocks) return mock();
  const res = await customMutator<Envelope<T>>(url, init);
  return res.data;
}

/** The API filters by one expert; Otto or several are narrowed here. */
function narrowsLocally(filters: TaskFilters) {
  const ids = filters.assignee_ids ?? [];
  return ids.length > 1 || ids.includes(OTTO_FILTER_ID);
}

function narrow<T extends Parameters<typeof matchesAssignees>[0]>(
  tasks: T[],
  filters: TaskFilters,
) {
  if (!narrowsLocally(filters)) return tasks;
  return tasks.filter((t) => matchesAssignees(t, filters.assignee_ids ?? []));
}

function json(method: string, body?: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  };
}

function toQuery(filters: TaskFilters, { withHidden = false } = {}) {
  const params = new URLSearchParams();
  if (filters.status?.length) params.set("status", filters.status.join(","));
  if (filters.status?.length || withHidden)
    params.set("include_hidden", "true");
  if (filters.priority?.length)
    params.set("priority", filters.priority.join(","));
  if (filters.assignee_ids?.length === 1 && !narrowsLocally(filters))
    params.set("assignee_id", filters.assignee_ids[0]);
  if (filters.due) params.set("due", filters.due);
  if (filters.mine) params.set("mine", "true");
  if (filters.parent_id) params.set("parent_id", filters.parent_id);
  const q = params.toString();
  return q ? `?${q}` : "";
}

export const tasksApi = {
  listWithCounts: (filters: TaskFilters) =>
    call<{ tasks: unknown[]; counts?: unknown }>(
      `/api/tasks${toQuery(filters, { withHidden: true })}`,
      json("GET"),
      () => ({ tasks: mockApi.list(filters) }),
    ).then((r) => {
      const all = r.tasks.map(toTask);
      if (!narrowsLocally(filters)) {
        return { tasks: all, counts: toTaskCounts(r.counts, all) };
      }
      const tasks = narrow(all, filters);
      return { tasks, counts: toTaskCounts(undefined, tasks) };
    }),
  list: (filters: TaskFilters) =>
    call<{ tasks: unknown[] }>(
      `/api/tasks${toQuery(filters)}`,
      json("GET"),
      () => ({ tasks: mockApi.list(filters) }),
    ).then((r) => narrow(r.tasks.map(toTask), filters)),
  needsYou: () =>
    call<{ tasks: unknown[] }>(`/api/tasks/needs-you`, json("GET"), () => ({
      tasks: mockApi.list({ status: ["blocked", "in_review"] }),
    })).then((r) => r.tasks.map(toTask)),
  get: (id: string) =>
    call<unknown>(`/api/tasks/${id}`, json("GET"), () => mockApi.get(id)).then(
      toTaskDetail,
    ),
  activity: (id: string) =>
    call<{ events?: unknown[]; activity?: unknown[] }>(
      `/api/tasks/${id}/activity`,
      json("GET"),
      () => ({ activity: mockApi.activity(id) }),
    ).then((r) => (r.events ?? r.activity ?? []).map(toActivity)),
  activityPage: (id: string, cursor: string | null) =>
    call<{ events: unknown[]; next_cursor?: string | null }>(
      `/api/tasks/${id}/activity?limit=${ACTIVITY_PAGE}${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`,
      json("GET"),
      () => ({ events: mockApi.activity(id), next_cursor: null }),
    ).then((r) => ({
      items: r.events.map(toActivity),
      nextCursor: r.next_cursor ?? null,
    })),
  summary: (id: string) =>
    call<TaskSummary>(`/api/tasks/${id}/summary`, json("GET"), () =>
      summaryFromTask(mockApi.get(id)),
    ),
  evidence: (id: string, evidenceId: string) =>
    call<EvidenceDetail | null>(
      `/api/tasks/${id}/evidence/${evidenceId}`,
      json("GET"),
      () => null,
    ),
  answerAsk: (id: string, answer: string, askId?: string) =>
    call<TaskSummary>(
      `/api/tasks/${id}/ask/answer`,
      json("POST", { answer, ask_id: askId }),
      () => summaryFromTask(mockApi.answer(id, answer)),
    ),
  answerApprovals: (id: string, decisions: CardDecision[]) =>
    call<TaskSummary>(
      `/api/tasks/${id}/approvals`,
      json("POST", { decisions }),
      () => summaryFromTask(mockApi.answer(id, "Answered the cards")),
    ),
  action: (id: string, action: string, body: TaskActionRequest = {}) =>
    call<TaskSummary>(
      `/api/tasks/${id}/actions/${action}`,
      json("POST", body),
      () => summaryFromTask(mockApi.get(id)),
    ),
  create: (input: CreateTaskInput) =>
    call<unknown>(`/api/tasks`, json("POST", input), () =>
      mockApi.create(input),
    ).then(toTask),
  update: (id: string, input: UpdateTaskInput) =>
    call<unknown>(`/api/tasks/${id}`, json("PATCH", input), () =>
      mockApi.update(id, input),
    ).then(toTask),
  rename: (id: string, title: string) =>
    send<unknown>(`/api/tasks/${id}`, json("PATCH", { title }), () =>
      mockApi.update(id, { title }),
    ).then(toTask),
  remove: (id: string) =>
    send<unknown>(`/api/tasks/${id}`, json("DELETE"), () =>
      mockApi.remove(id),
    ).then(() => undefined),
  start: (id: string) =>
    call<unknown>(`/api/tasks/${id}/start`, json("POST", {}), () =>
      mockApi.update(id, { status: "in_progress" }),
    ).then(toTask),
  comment: (id: string, body: string) =>
    call<unknown>(`/api/tasks/${id}/comments`, json("POST", { body }), () =>
      mockApi.comment(id, body),
    ).then(toActivity),
  answer: (id: string, answer: string, inputId?: string) =>
    call<unknown>(
      `/api/tasks/${id}/answer`,
      json("POST", { answer, input_id: inputId }),
      () => mockApi.answer(id, answer),
    ).then(toTask),
  experts: () =>
    call<{ experts: unknown[] }>(`/api/tasks/experts`, json("GET"), () => ({
      experts: mockApi.experts(),
    })).then((r) => r.experts.map(toExpert)),
  getPlan: (planId: string) =>
    call<unknown>(`/api/tasks/plans/${planId}`, json("GET"), () => null).then(
      toPlan,
    ),
  approvePlan: (plan: TaskPlan) =>
    call<{ parent_task?: unknown } & Record<string, unknown>>(
      `/api/tasks/plans/${plan.id}/confirm`,
      json("POST", {}),
      () => mockApi.approvePlan(plan) as unknown as Record<string, unknown>,
    ).then((r) => toTask(r.parent_task ?? r)),
};
