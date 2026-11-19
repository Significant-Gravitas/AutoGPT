import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getListExpertsResponseMock200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
import { TaskTab } from "../TaskTab";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const TASK_URL = "/api/proxy/api/tasks/task-1";
const [leoMock] = getListExpertsResponseMock200();
const LEO = { ...leoMock, id: "expert-leo", name: "Leo" };

let task: Task;
let calls: { url: string; method: string; body: unknown }[];

function setTask(status: TaskStatus, extra: Partial<Task> = {}) {
  task = { ...makeTask("task-1", "Plan the launch", status), ...extra };
}

async function bodyOf(request: Request) {
  const text = await request.text();
  return text ? JSON.parse(text) : null;
}

/** The task's record as it stands, and the run, cancel and edit routes,
 *  each moving the task the way the API does. */
function serveTask() {
  server.use(
    http.get(`${TASK_URL}/record`, () =>
      HttpResponse.json({ task, activity: [], artifacts: [], evidence: [] }),
    ),
    http.post(`${TASK_URL}/run`, async ({ request }) => {
      calls.push({ url: "run", method: "POST", body: await bodyOf(request) });
      setTask("IN_PROGRESS");
      return HttpResponse.json(task);
    }),
    http.post(`${TASK_URL}/cancel`, () => {
      calls.push({ url: "cancel", method: "POST", body: null });
      setTask("CANCELLED");
      return HttpResponse.json(task);
    }),
    http.patch(TASK_URL, async ({ request }) => {
      const body = await bodyOf(request);
      calls.push({ url: "update", method: "PATCH", body });
      task = { ...task, ...body };
      return HttpResponse.json(task);
    }),
  );
}

function renderTab() {
  return render(
    <TaskTab
      taskID="task-1"
      sessionID={null}
      experts={[LEO]}
      onOpenArtifact={vi.fn()}
    />,
  );
}

beforeEach(() => {
  calls = [];
  mockToast.mockClear();
  serveTask();
});

describe("Task tab controls", () => {
  it("starts a task that is to do", async () => {
    setTask("TODO");
    renderTab();

    await userEvent.click(await screen.findByRole("button", { name: "Start" }));

    expect(await screen.findByRole("button", { name: "Stop" })).toBeDefined();
    expect(calls).toEqual([{ url: "run", method: "POST", body: null }]);
  });

  it("stops a running task after asking", async () => {
    setTask("IN_PROGRESS");
    renderTab();

    await userEvent.click(await screen.findByRole("button", { name: "Stop" }));
    const dialog = await screen.findByRole("dialog", {
      name: "Stop this task?",
    });
    expect(calls).toEqual([]);
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Stop task" }),
    );

    expect(
      await screen.findByRole("button", { name: "Try again" }),
    ).toBeDefined();
    expect(calls.map((call) => call.url)).toEqual(["cancel"]);
  });

  it("tries a failed task again with what to do differently", async () => {
    setTask("FAILED", { failure_reason: "The work ran out of time." });
    renderTab();

    await userEvent.click(
      await screen.findByRole("button", { name: "Try again" }),
    );
    const form = await screen.findByRole("form", {
      name: "Try the task again",
    });
    const field = within(form).getByLabelText("Anything to do differently?");
    await expectTypingKept(
      field as HTMLTextAreaElement,
      "Use the Q3 numbers 日本",
    );
    await userEvent.click(
      within(form).getByRole("button", { name: "Try again" }),
    );

    expect(await screen.findByRole("button", { name: "Stop" })).toBeDefined();
    expect(calls).toEqual([
      {
        url: "run",
        method: "POST",
        body: { instructions: "Use the Q3 numbers 日本" },
      },
    ]);
    expect(screen.queryByRole("form", { name: "Try the task again" })).toBe(
      null,
    );
  });

  it("runs a finished task again", async () => {
    setTask("DONE", { result: "Plan written." });
    renderTab();

    await userEvent.click(
      await screen.findByRole("button", { name: "Run again" }),
    );

    expect(await screen.findByRole("button", { name: "Stop" })).toBeDefined();
    expect(calls).toEqual([{ url: "run", method: "POST", body: null }]);
  });

  it("shows why a start was refused", async () => {
    setTask("TODO");
    server.use(
      http.post(`${TASK_URL}/run`, () =>
        HttpResponse.json(
          { detail: "Too many tasks are running." },
          { status: 429 },
        ),
      ),
    );
    renderTab();

    await userEvent.click(await screen.findByRole("button", { name: "Start" }));

    await vi.waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't start the task",
          description: "Too many tasks are running.",
          variant: "destructive",
        }),
      ),
    );
    expect(screen.getByRole("button", { name: "Start" })).toBeDefined();
  });
});
