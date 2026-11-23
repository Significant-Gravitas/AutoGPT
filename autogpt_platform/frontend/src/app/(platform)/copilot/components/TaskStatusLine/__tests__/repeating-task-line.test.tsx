import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { startedTaskOf } from "../helpers";
import { TaskStatusLine } from "../TaskStatusLine";

const NOW = new Date(2026, 10, 15, 10, 0, 0).getTime();
const NEXT_RUN = new Date(NOW + 4 * 60_000);
const TASK_URL = "/api/proxy/api/tasks/task-1";
const REPEAT = {
  crons: ["*/4 * * * *"],
  timezone: "UTC",
  label: "Every 4 minutes",
  paused: false,
};

let task: Task;
let calls: string[];

/** The start_task result Otto hands back for a repeating task. */
const RESULT = {
  type: "task_started",
  in_background: true,
  start_at: NEXT_RUN.toISOString(),
  task: {
    id: "task-1",
    title: "Research AutoGPT",
    status: "TODO",
    owner_expert_id: null,
    start_at: NEXT_RUN.toISOString(),
    missed_start: false,
    repeat: REPEAT,
  },
};

function serveTask() {
  server.use(
    getListExpertsMockHandler200([]),
    http.get(TASK_URL, () => HttpResponse.json(task)),
    http.post(`${TASK_URL}/run`, () => {
      calls.push("run");
      task = { ...task, status: "IN_PROGRESS", start_at: null };
      return HttpResponse.json(task);
    }),
    http.post(`${TASK_URL}/repeat/pause`, () => {
      calls.push("pause");
      task = {
        ...task,
        start_at: null,
        repeat: { ...REPEAT, paused: true },
      };
      return HttpResponse.json(task);
    }),
    http.post(`${TASK_URL}/repeat/resume`, () => {
      calls.push("resume");
      task = { ...task, start_at: NEXT_RUN, repeat: REPEAT };
      return HttpResponse.json(task);
    }),
    http.delete(`${TASK_URL}/repeat`, () => {
      calls.push("stop-repeating");
      task = { ...task, repeat: null };
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
    start_at: NEXT_RUN,
    repeat: REPEAT,
  };
  serveTask();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Repeating task card in chat", () => {
  it("reads the rhythm from the start_task result", () => {
    expect(startedTaskOf(RESULT)?.repeat).toEqual({
      label: "Every 4 minutes",
      paused: false,
    });
  });

  it("shows the rhythm and the next run, with Run now and Pause", async () => {
    renderLine();

    expect(await screen.findByTestId("repeating-task-line")).toBeDefined();
    expect(
      screen.getByText("Every 4 minutes · next run in 4 min"),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Run now" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Pause" })).toBeDefined();
  });

  it("pauses it, then reads Paused and offers Resume", async () => {
    renderLine();

    await userEvent.click(await screen.findByRole("button", { name: "Pause" }));

    expect(await screen.findByText("Every 4 minutes · Paused")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Resume" }));
    expect(
      await screen.findByText("Every 4 minutes · next run in 4 min"),
    ).toBeDefined();
    expect(calls).toEqual(["pause", "resume"]);
  });

  it("says why when three failed runs paused it", async () => {
    task = {
      ...task,
      start_at: null,
      status_text: "Paused: the last 3 runs failed",
      repeat: { ...REPEAT, paused: true, failed_in_a_row: 3 },
    };
    renderLine();

    expect(
      await screen.findByText(
        "Every 4 minutes · Paused: the last 3 runs failed",
      ),
    ).toBeDefined();
  });

  it("stops repeating from its menu", async () => {
    renderLine();

    await userEvent.click(
      await screen.findByRole("button", { name: "More for Research AutoGPT" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitem", { name: "Stop repeating" }),
    );

    await waitFor(() => expect(calls).toEqual(["stop-repeating"]));
  });

  it("shows the run working, then comes back for the next run", async () => {
    renderLine();
    await userEvent.click(
      await screen.findByRole("button", { name: "Run now" }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("task-status-badge").dataset.state).toBe(
        "working",
      ),
    );

    // The run ends: the task waits again for its next run.
    task = { ...task, status: "TODO", start_at: NEXT_RUN };
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(await screen.findByTestId("repeating-task-line")).toBeDefined();
    expect(calls).toEqual(["run"]);
  });
});
