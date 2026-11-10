import {
  getListExpertsMockHandler200,
  getListExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetTaskMockHandler200,
  getGetTaskResponseMock200,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import type { StartedTask } from "../helpers";
import { TaskStatusLine } from "../TaskStatusLine";

const STARTED: StartedTask = {
  id: "task-1",
  title: "Research Notion's competitors",
  status: "IN_PROGRESS",
  ownerExpertID: null,
  startedAt: "2026-11-10T09:59:00Z",
  link: "/copilot?sessionId=child-1",
};

function taskNow(overrides: Partial<Task>) {
  return getGetTaskMockHandler200(
    getGetTaskResponseMock200({
      id: STARTED.id,
      title: STARTED.title,
      owner_expert_id: null,
      started_at: new Date("2026-11-10T09:59:00Z"),
      completed_at: null,
      ...overrides,
    }),
  );
}

function badge() {
  return screen.getByTestId("task-status-badge").getAttribute("data-state");
}

describe("TaskStatusLine", () => {
  it("says Otto is working and links to the work while the task runs", async () => {
    server.use(taskNow({ status: "IN_PROGRESS" }));

    render(<TaskStatusLine task={STARTED} />);

    expect(
      screen.getByText(/is working on Research Notion's competitors\./),
    ).toBeDefined();
    expect(screen.getByText("Otto")).toBeDefined();
    expect(badge()).toBe("working");
    const link = screen.getByRole("link", { name: "See the work" });
    expect(link.getAttribute("href")).toBe("/copilot?sessionId=child-1");
  });

  it("shows the task finished, with how long it took, once the task is done", async () => {
    server.use(
      taskNow({
        status: "DONE",
        completed_at: new Date("2026-11-10T10:03:12Z"),
      }),
    );

    render(<TaskStatusLine task={STARTED} />);

    expect(
      await screen.findByText(/finished Research Notion's competitors\./),
    ).toBeDefined();
    expect(screen.getByText("4m 12s")).toBeDefined();
    expect(badge()).toBe("done");
  });

  it("says the task waits on the user, with no link, when it needs input", async () => {
    server.use(taskNow({ status: "NEEDS_INPUT" }));

    render(<TaskStatusLine task={STARTED} />);

    expect(
      await screen.findByText(/needs your answer to carry on with/),
    ).toBeDefined();
    expect(screen.queryByRole("link", { name: "See the work" })).toBeNull();
  });

  it("names the expert who owns the task", async () => {
    server.use(
      taskNow({ status: "IN_PROGRESS", owner_expert_id: "exp-1" }),
      getListExpertsMockHandler200([expertNamed("exp-1", "Harper")]),
    );

    render(<TaskStatusLine task={{ ...STARTED, ownerExpertID: "exp-1" }} />);

    await waitFor(() => expect(screen.getByText("Harper")).toBeDefined());
  });
});

function expertNamed(id: string, name: string) {
  const [expert] = getListExpertsResponseMock200();
  return { ...expert, id, name, avatar_url: null };
}
