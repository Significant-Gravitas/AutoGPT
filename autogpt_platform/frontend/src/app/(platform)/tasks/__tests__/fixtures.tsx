import {
  getListExpertsMockHandler200,
  getListExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getArchiveTaskMockHandler,
  getCreateTaskMockHandler,
  getGetTaskMockHandler,
  getListTasksMockHandler,
  getUpdateTaskMockHandler,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskCreateRequest } from "@/app/api/__generated__/models/taskCreateRequest";
import type { TaskUpdateRequest } from "@/app/api/__generated__/models/taskUpdateRequest";
import { InsetSideSlot } from "@/components/layout/InsetSidePanel/InsetSidePanel";
import { server } from "@/mocks/mock-server";
import { render } from "@/tests/integrations/test-utils";
import { HttpResponse } from "msw";
import TasksPage from "../page";

export const MAYA: Expert = {
  ...getListExpertsResponseMock200()[0],
  id: "expert-maya",
  name: "Maya",
  avatar_url: null,
};

export function makeTask(overrides: Partial<Task> & { id: string }): Task {
  return {
    user_id: "user-1",
    title: `Task ${overrides.id}`,
    description: null,
    status: "TODO",
    owner_expert_id: null,
    created_by: "USER",
    created_by_expert_id: null,
    started_at: null,
    completed_at: null,
    is_archived: false,
    session_ids: [],
    created_at: new Date("2026-11-10T09:00:00Z"),
    updated_at: new Date("2026-11-10T09:00:00Z"),
    ...overrides,
  };
}

/** A tiny in-memory task API: list, get, create, update and archive all
 *  read and write the same tasks, and every request is recorded. */
export function serveTasks(initial: Task[], experts: Expert[] = [MAYA]) {
  const api = {
    tasks: [...initial],
    listCalls: 0,
    created: [] as TaskCreateRequest[],
    updates: [] as { id: string; body: TaskUpdateRequest }[],
    archived: [] as string[],
  };

  server.use(
    getListExpertsMockHandler200(experts),
    getListTasksMockHandler(() => {
      api.listCalls += 1;
      return { tasks: api.tasks };
    }),
    getGetTaskMockHandler((info) => {
      const task = api.tasks.find((t) => t.id === info.params.taskId);
      if (!task)
        throw HttpResponse.json({ detail: "Not found" }, { status: 404 });
      return task;
    }),
    getCreateTaskMockHandler(async (info) => {
      const body = (await info.request.json()) as TaskCreateRequest;
      api.created.push(body);
      const task = makeTask({
        id: `new-${api.created.length}`,
        title: body.title,
        description: body.description ?? null,
        owner_expert_id: body.owner_expert_id ?? null,
        updated_at: new Date("2026-11-10T11:00:00Z"),
      });
      api.tasks = [...api.tasks, task];
      return task;
    }),
    getUpdateTaskMockHandler(async (info) => {
      const id = String(info.params.taskId);
      const body = (await info.request.json()) as TaskUpdateRequest;
      api.updates.push({ id, body });
      api.tasks = api.tasks.map((t) =>
        t.id === id
          ? {
              ...t,
              title: body.title ?? t.title,
              status: body.status ?? t.status,
            }
          : t,
      );
      return api.tasks.find((t) => t.id === id) ?? makeTask({ id });
    }),
    getArchiveTaskMockHandler((info) => {
      const id = String(info.params.taskId);
      api.archived.push(id);
      api.tasks = api.tasks.filter((t) => t.id !== id);
    }),
  );

  return api;
}

export function renderTasksPage() {
  return render(
    <>
      <InsetSideSlot />
      <TasksPage />
    </>,
  );
}
