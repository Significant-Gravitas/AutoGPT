import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { startedTaskOf, uniqueTasks } from "../helpers";
import { TaskStatusLine } from "../TaskStatusLine";

const NOW = new Date(2026, 10, 15, 10, 0, 0).getTime();
const START_AT = new Date(NOW + 4 * 60_000);
const TASK_URL = "/api/proxy/api/tasks/task-1";

let task: Task;
let calls: string[];

/** The start_task result Otto's tool hands back for a task set for later. */
const RESULT = {
  type: "task_started",
  in_background: true,
  start_at: START_AT.toISOString(),
  task: {
    id: "task-1",
    title: "Research AutoGPT",
    status: "TODO",
    owner_expert_id: null,
    start_at: START_AT.toISOString(),
    missed_start: false,
  },
};

function serveTask() {
  server.use(
    getListExpertsMockHandler200([]),
    http.get(TASK_URL, () => HttpResponse.json(task)),
    http.post(`${TASK_URL}/run`, () => {
      calls.push("run");
      task = {
        ...task,
        status: "IN_PROGRESS",
        start_at: null,
        started_at: new Date(NOW),
      };
      return HttpResponse.json(task);
    }),
    http.post(`${TASK_URL}/cancel`, () => {
      calls.push("cancel");
      task = { ...task, status: "CANCELLED" };
      return HttpResponse.json(task);
    }),
  );
}

function renderLine() {
  const started = startedTaskOf(RESULT);
  if (!started) throw new Error("The result should read as a started task");
  return render(<TaskStatusLine task={started} />);
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
  calls = [];
  task = {
    ...makeTask("task-1", "Research AutoGPT", "TODO"),
    start_at: START_AT,
  };
  serveTask();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Scheduled task card in chat", () => {
  it("reads the start time from the start_task result", () => {
    const started = startedTaskOf(RESULT);
    expect(started?.status).toBe("TODO");
    expect(started?.startAt).toBe(START_AT.toISOString());
    expect(uniqueTasks([started!, started!])).toHaveLength(1);
  });

  it("shows the countdown with the exact time, and Start now starts it", async () => {
    renderLine();

    expect(await screen.findByTestId("scheduled-task-line")).toBeDefined();
    expect(screen.getByText(/^Starts in 4 min · /)).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Start now" }));

    await waitFor(() =>
      expect(screen.queryByTestId("scheduled-task-line")).toBeNull(),
    );
    expect(calls).toEqual(["run"]);
    expect(screen.getByTestId("task-status-badge").dataset.state).toBe(
      "working",
    );
  });

  it("cancels it, and then says it was cancelled", async () => {
    renderLine();

    await userEvent.click(
      await screen.findByRole("button", { name: "Cancel" }),
    );

    expect(await screen.findByText("Cancelled")).toBeDefined();
    expect(calls).toEqual(["cancel"]);
    expect(screen.queryByTestId("scheduled-task-line")).toBeNull();
  });

  it("offers Start now on a task that missed its start time", async () => {
    task = { ...task, start_at: null, missed_start: true };
    renderLine();

    expect(await screen.findByText("Missed its start time")).toBeDefined();
    expect(screen.getByRole("button", { name: "Start now" })).toBeDefined();
  });
});
