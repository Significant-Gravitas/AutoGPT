import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskRun } from "@/app/api/__generated__/models/taskRun";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TaskTab } from "../TaskTab";

const NOW = new Date(2026, 10, 15, 10, 0, 0).getTime();
const TASK_URL = "/api/proxy/api/tasks/task-1";
const REPEAT = {
  crons: ["0 8 * * *"],
  timezone: "UTC",
  label: "Every day at 08:00",
  paused: false,
};

let task: Task;
let runs: TaskRun[];
let calls: { url: string; body: unknown }[];

async function bodyOf(request: Request) {
  const text = await request.text();
  return text ? JSON.parse(text) : null;
}

function at(hoursAgo: number) {
  return new Date(NOW - hoursAgo * 60 * 60_000);
}

function serveTask() {
  server.use(
    http.get(TASK_URL, () => HttpResponse.json(task)),
    http.get(`${TASK_URL}/record`, () =>
      HttpResponse.json({
        task,
        activity: [],
        artifacts: [],
        evidence: [],
        runs,
      }),
    ),
    http.post(`${TASK_URL}/cancel`, async ({ request }) => {
      calls.push({ url: "cancel", body: await bodyOf(request) });
      task = { ...task, status: "TODO", start_at: new Date(NOW + 3_600_000) };
      return HttpResponse.json(task);
    }),
    http.post(`${TASK_URL}/repeat/pause`, () => {
      calls.push({ url: "pause", body: null });
      task = { ...task, start_at: null, repeat: { ...REPEAT, paused: true } };
      return HttpResponse.json(task);
    }),
  );
}

function renderTab() {
  return render(
    <TaskTab
      taskID="task-1"
      sessionID={null}
      experts={[]}
      onOpenArtifact={vi.fn()}
    />,
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
  calls = [];
  task = {
    ...makeTask("task-1", "Morning brief", "TODO"),
    start_at: new Date(NOW + 22 * 3_600_000),
    result: "Three new signups overnight.",
    repeat: REPEAT,
  };
  runs = [
    {
      chat_session_id: "run-2",
      started_at: at(2),
      ended_at: at(1.9),
      outcome: "done",
      result: "Three new signups overnight.",
    },
    {
      chat_session_id: "run-1",
      started_at: at(26),
      ended_at: at(25.9),
      outcome: "failed",
      failure_reason: "The work ran out of time.",
    },
  ];
  serveTask();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Task tab for a repeating task", () => {
  it("shows its rhythm, Run now and Pause, and no result block of its own", async () => {
    renderTab();

    expect(await screen.findByText("Every day at 08:00")).toBeDefined();
    expect(screen.getByText("Repeats")).toBeDefined();
    expect(screen.getByRole("button", { name: "Run now" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Pause" })).toBeDefined();
    expect(screen.queryByRole("region", { name: "Last result" })).toBeNull();
  });

  it("lists each run newest first, and a click opens its result in place", async () => {
    renderTab();

    const group = await screen.findByRole("region", { name: "Runs" });
    const rows = within(group).getAllByRole("button");
    expect(within(rows[0]).getByText("Done")).toBeDefined();
    expect(within(rows[0]).getByText(/^Today, /)).toBeDefined();
    expect(within(rows[1]).getByText("Failed")).toBeDefined();
    expect(within(group).queryByRole("link")).toBeNull();

    await userEvent.click(rows[1]);

    expect(
      await within(group).findByText("The work ran out of time."),
    ).toBeDefined();
    expect(
      within(group)
        .getByRole("link", { name: /Open the chat it ran in/ })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=run-1");
  });

  it("pausing it reads Paused", async () => {
    renderTab();

    await userEvent.click(await screen.findByRole("button", { name: "Pause" }));

    expect(await screen.findByRole("button", { name: "Resume" })).toBeDefined();
    expect(calls).toEqual([{ url: "pause", body: null }]);
  });

  it("Stop on a running repeating task stops only this run", async () => {
    task = { ...task, status: "IN_PROGRESS", start_at: null };
    renderTab();

    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Stop run" }),
    );

    await vi.waitFor(() =>
      expect(calls).toEqual([{ url: "cancel", body: { this_run_only: true } }]),
    );
  });
});
