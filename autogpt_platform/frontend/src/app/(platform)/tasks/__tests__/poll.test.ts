import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { describe, expect, it } from "vitest";
import { POLL_MS, pollInterval } from "../helpers";

const NOW = new Date(2026, 10, 19, 10, 0, 0).getTime();
const MIN = 60_000;

function task(status: TaskStatus, extra: Partial<Task> = {}): Task {
  return { ...makeTask("t", "Task", status), ...extra };
}

function poll(...tasks: Task[]) {
  return pollInterval(tasks, NOW);
}

describe("Task list polling", () => {
  it("does not poll for a plain to-do saved for later, or for finished work", () => {
    expect(poll(task("TODO"))).toBe(false);
    expect(poll(task("DONE"), task("FAILED"), task("CANCELLED"))).toBe(false);
    expect(pollInterval(undefined, NOW)).toBe(false);
  });

  it("polls while a task is being worked or waits on the user", () => {
    expect(poll(task("TODO"), task("IN_PROGRESS"))).toBe(POLL_MS);
    expect(poll(task("NEEDS_INPUT"))).toBe(POLL_MS);
  });

  it("polls for a scheduled or repeating task only when its start is near", () => {
    expect(poll(task("TODO", { start_at: new Date(NOW + 2 * MIN) }))).toBe(
      POLL_MS,
    );
    expect(poll(task("TODO", { start_at: new Date(NOW - 1 * MIN) }))).toBe(
      POLL_MS,
    );
    expect(poll(task("TODO", { start_at: new Date(NOW + 3 * 60 * MIN) }))).toBe(
      false,
    );
    const repeat = { crons: ["0 10 * * *"], label: "Every day at 10:00" };
    expect(
      poll(task("TODO", { repeat, start_at: new Date(NOW + 5 * MIN) })),
    ).toBe(POLL_MS);
    expect(
      poll(task("TODO", { repeat, start_at: new Date(NOW + 20 * 60 * MIN) })),
    ).toBe(false);
  });

  it("polls for a watching task only while an event waits to start it", () => {
    const watch = {
      webhook_id: "wh-1",
      provider: "github",
      trigger_block_id: "block-1",
      line: "PR opened on acme/web",
    };
    const repeat = { label: watch.line, watch };
    expect(poll(task("TODO", { repeat }))).toBe(false);
    expect(poll(task("TODO", { repeat, events_waiting: 1 }))).toBe(POLL_MS);
  });

  it("does not poll for a paused or missed task", () => {
    const paused = { label: "Every day", paused: true };
    expect(
      poll(task("TODO", { repeat: paused, start_at: new Date(NOW + MIN) })),
    ).toBe(false);
    expect(
      poll(task("TODO", { missed_start: true, start_at: new Date(NOW - MIN) })),
    ).toBe(false);
  });
});
