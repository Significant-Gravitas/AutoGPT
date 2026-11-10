import { describe, expect, it } from "vitest";
import {
  clock,
  duration,
  isSettled,
  secondsBetween,
  startedTaskOf,
  taskLineStateOf,
} from "../helpers";

const OUTPUT = {
  type: "task_started",
  task: {
    id: "task-1",
    title: "Research Notion's competitors",
    status: "IN_PROGRESS",
    owner_expert_id: null,
    started_at: "2026-11-10T09:59:00Z",
  },
  in_background: true,
  worker_session_link: "/copilot?sessionId=child-1",
};

describe("startedTaskOf", () => {
  it("reads the task and the chat doing the work from a start_task result", () => {
    expect(startedTaskOf(OUTPUT)).toEqual({
      id: "task-1",
      title: "Research Notion's competitors",
      status: "IN_PROGRESS",
      ownerExpertID: null,
      startedAt: "2026-11-10T09:59:00Z",
      link: "/copilot?sessionId=child-1",
    });
    expect(startedTaskOf(JSON.stringify(OUTPUT))?.id).toBe("task-1");
  });

  it("ignores every other tool result", () => {
    expect(
      startedTaskOf({ type: "task_created", task: OUTPUT.task }),
    ).toBeNull();
    expect(startedTaskOf({ type: "task_started" })).toBeNull();
    expect(startedTaskOf("not json")).toBeNull();
  });
});

describe("task status", () => {
  it.each([
    ["TODO", "working", false],
    ["IN_PROGRESS", "working", false],
    ["NEEDS_INPUT", "needs_input", false],
    ["DONE", "done", true],
    ["FAILED", "failed", true],
    ["CANCELLED", "stopped", true],
  ] as const)("%s reads as %s", (status, state, settled) => {
    expect(taskLineStateOf(status)).toBe(state);
    expect(isSettled(status)).toBe(settled);
  });
});

describe("time", () => {
  it("counts whole seconds and never goes negative", () => {
    const start = "2026-11-10T09:59:00Z";
    expect(secondsBetween(start, "2026-11-10T10:00:24Z")).toBe(84);
    expect(secondsBetween(start, "2026-11-10T09:58:00Z")).toBe(0);
    expect(secondsBetween(null, start)).toBeNull();
    expect(secondsBetween(start, null)).toBeNull();
  });

  it("shows a running clock and a finished duration", () => {
    expect(clock(84)).toBe("1:24");
    expect(clock(3725)).toBe("1:02:05");
    expect(duration(45)).toBe("45s");
    expect(duration(252)).toBe("4m 12s");
    expect(duration(3780)).toBe("1h 3m");
  });
});
