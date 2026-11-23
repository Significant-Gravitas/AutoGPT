import {
  getGetTaskMockHandler200,
  getGetTaskResponseMock200,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import type { StartedTask } from "../helpers";
import { TaskStatusLine } from "../TaskStatusLine";

const STARTED: StartedTask = {
  id: "task-1",
  title: "Find design partners",
  status: "IN_PROGRESS",
  ownerExpertID: null,
  startedAt: "2026-11-19T09:59:00Z",
  link: null,
};

function serveTask(overrides: Partial<Task>) {
  server.use(
    getGetTaskMockHandler200(
      getGetTaskResponseMock200({
        id: STARTED.id,
        title: STARTED.title,
        owner_expert_id: null,
        started_at: new Date("2026-11-19T09:59:00Z"),
        completed_at: null,
        repeat: null,
        start_at: null,
        missed_start: false,
        subtask_count: 0,
        open_ask_count: 0,
        review: null,
        ...overrides,
      }),
    ),
  );
}

function badge() {
  return screen.getByTestId("task-status-badge").getAttribute("data-state");
}

describe("the chat's task card", () => {
  it("shows a run waiting for a free slot as Queued, not Working", async () => {
    serveTask({
      status: "IN_PROGRESS",
      is_queued: true,
      status_text: "Queued: starts when a slot frees",
    });
    render(<TaskStatusLine task={STARTED} />);

    expect(
      await screen.findByText("Queued · Starts when a slot frees"),
    ).toBeDefined();
    expect(badge()).toBe("queued");
    expect(screen.queryByText(/Working/)).toBeNull();
  });

  it("shows a task that did not deliver as Failed with its reason", async () => {
    serveTask({
      status: "FAILED",
      is_queued: false,
      failure_reason: "No web search was available.",
      result: "No web search was available.",
      completed_at: new Date("2026-11-19T10:03:00Z"),
    });
    render(<TaskStatusLine task={STARTED} />);

    expect(
      await screen.findByText("Failed: No web search was available."),
    ).toBeDefined();
    expect(badge()).toBe("failed");
  });

  it("still shows an old task that settled as Done", async () => {
    serveTask({
      status: "DONE",
      is_queued: undefined,
      failure_reason: null,
      completed_at: new Date("2026-11-19T10:03:00Z"),
    });
    render(<TaskStatusLine task={STARTED} />);

    expect(await screen.findByText("Done")).toBeDefined();
    expect(badge()).toBe("done");
  });
});
