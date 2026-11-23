import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { TaskTab } from "../TaskTab";

const TASK_URL = "/api/proxy/api/tasks/task-1";
let task: Task;

function serve(extra: Partial<Task>) {
  task = { ...makeTask("task-1", "Find design partners", "TODO"), ...extra };
  server.use(
    http.get(TASK_URL, () => HttpResponse.json(task)),
    http.get(`${TASK_URL}/record`, () =>
      HttpResponse.json({ task, activity: [], artifacts: [], evidence: [] }),
    ),
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

describe("Task tab status", () => {
  it("shows a run waiting for a free slot as Queued", async () => {
    serve({
      status: "IN_PROGRESS",
      is_queued: true,
      status_text: "Queued: starts when a slot frees",
    });
    renderTab();

    const status = await screen.findByRole("region", { name: "Status" });
    expect(await within(status).findByText("Queued")).toBeDefined();
    expect(within(status).getByText("Starts when a slot frees")).toBeDefined();
    expect(within(status).queryByText("Working")).toBeNull();
  });

  it("shows a task that did not deliver as Failed, with what went wrong", async () => {
    serve({
      status: "FAILED",
      failure_reason: "No web search was available.",
      result: "No web search was available.",
    });
    renderTab();

    const status = await screen.findByRole("region", { name: "Status" });
    expect(await within(status).findByText("Failed")).toBeDefined();
    const wrong = await screen.findByRole("region", {
      name: "What went wrong",
    });
    expect(
      within(wrong).getByText("No web search was available."),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Accept" })).toBeNull();
  });
});
