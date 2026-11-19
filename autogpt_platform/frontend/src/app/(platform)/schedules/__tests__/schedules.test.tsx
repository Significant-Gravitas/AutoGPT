import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";

import { getGetV1ListGraphExecutionsMockHandler200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV1ListExecutionSchedulesForAUserMockHandler200,
  getListUpcomingScheduleRunsMockHandler200,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import {
  getGetTaskRecordMockHandler200,
  getListTasksMockHandler200,
  getPauseTaskRepeatMockHandler200,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";

import SchedulesPage from "../page";

const HOUR = 3_600_000;
/** A schedule's row – its run markers on the chart carry the same name. */
const ROW = { selector: "button[data-id]" };
const NOW = Date.now();

function task(overrides: Partial<Task>): Task {
  return {
    id: "task-1",
    user_id: "user-1",
    title: "Weekly report",
    status: "TODO",
    created_by: "USER",
    created_at: new Date(NOW - 48 * HOUR),
    updated_at: new Date(NOW - HOUR),
    session_ids: ["chat-1"],
    work_session_id: "chat-1",
    ...overrides,
  };
}

const REPEATING = task({
  goal: "Sum up the week's sales for the team.",
  start_at: new Date(NOW + 2 * HOUR),
  repeat: {
    crons: ["0 9 * * 1"],
    timezone: "UTC",
    label: "Every Monday at 9:00",
    last_run: "failed",
  },
  started_at: new Date(NOW - 30 * HOUR),
  failure_reason: "Shopify did not answer",
});

const ONCE = task({
  id: "task-2",
  title: "Send the board pack",
  start_at: new Date(NOW + 5 * HOUR),
});

const WORKFLOW = {
  id: "sched-1",
  name: "Morning digest",
  user_id: "user-1",
  graph_id: "graph-1",
  graph_version: 3,
  agent_name: "Digest",
  cron: "0 8 * * *",
  input_data: {},
  next_run_time: new Date(NOW + 3 * HOUR).toISOString(),
  timezone: "UTC",
};

function serveSchedules() {
  const paused = vi.fn();
  server.use(
    getListTasksMockHandler200({ tasks: [REPEATING, ONCE] }),
    getGetV1ListExecutionSchedulesForAUserMockHandler200([WORKFLOW]),
    getListUpcomingScheduleRunsMockHandler200({
      schedules: [
        { kind: "task", id: "task-1", runs: [new Date(NOW + 2 * HOUR)] },
        { kind: "workflow", id: "sched-1", runs: [new Date(NOW + 3 * HOUR)] },
      ],
    }),
    getGetTaskRecordMockHandler200({ task: REPEATING, runs: [] }),
    getGetV1ListGraphExecutionsMockHandler200({
      executions: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 1,
        page_size: 10,
      },
    }),
    getPauseTaskRepeatMockHandler200(() => {
      paused();
      return { ...REPEATING, repeat: { ...REPEATING.repeat!, paused: true } };
    }),
  );
  return { paused };
}

describe("Schedules page", () => {
  it("lists task and workflow schedules together", async () => {
    serveSchedules();
    render(<SchedulesPage />);

    expect(
      await screen.findByLabelText(/^Weekly report, Every Monday/, ROW),
    ).toBeDefined();
    expect(
      screen.getByLabelText(/^Send the board pack, Once/, ROW),
    ).toBeDefined();
    expect(screen.getByLabelText(/^Morning digest, /, ROW)).toBeDefined();
    expect(screen.getByText("3 / 3")).toBeDefined();
  });

  it("pauses a repeating task, and offers no pause for a workflow", async () => {
    const { paused } = serveSchedules();
    render(<SchedulesPage />);

    fireEvent.click(
      await screen.findByLabelText("Weekly report: Pause", {
        selector: "button",
      }),
    );
    await vi.waitFor(() => expect(paused).toHaveBeenCalledTimes(1));

    expect(
      screen.queryByLabelText(/^Morning digest: Pause/, { selector: "button" }),
    ).toBeNull();
  });

  it("opens a task with its brief, last run and a way to its chat", async () => {
    serveSchedules();
    render(<SchedulesPage />);

    fireEvent.click(
      await screen.findByLabelText(/^Weekly report, Every Monday/, ROW),
    );

    expect(
      await screen.findByText("Sum up the week's sales for the team."),
    ).toBeDefined();
    expect(screen.getByText("Shopify did not answer")).toBeDefined();
    expect(screen.getByText("Open in chat")).toBeDefined();
  });

  it("opens a workflow schedule with Run now and Delete but no Pause", async () => {
    serveSchedules();
    render(<SchedulesPage />);

    fireEvent.click(await screen.findByLabelText(/^Morning digest, /, ROW));

    expect(await screen.findByText("Runs the workflow Digest.")).toBeDefined();
    expect(screen.getByText("Run now")).toBeDefined();
    expect(screen.getByText("Delete")).toBeDefined();
    expect(screen.queryByText("Pause")).toBeNull();
  });
});
