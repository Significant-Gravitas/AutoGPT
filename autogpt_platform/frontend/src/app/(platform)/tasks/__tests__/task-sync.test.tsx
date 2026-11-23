import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TaskList } from "../components/TaskList/TaskList";
import { isTaskGone } from "../taskErrors";
import { useTaskSync } from "../taskSync";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const TITLE = "Plan the offsite";
let reads: number;

function Harness() {
  useTaskSync();
  const tasks = useListTasks(
    { limit: 100 },
    { query: { select: (res) => okData(res)?.tasks ?? [] } },
  );
  return (
    <TaskList
      tasks={tasks.data ?? []}
      experts={[]}
      selectedID={null}
      onOpen={vi.fn()}
    />
  );
}

async function rename(next: string) {
  await userEvent.click(
    await screen.findByRole("button", { name: `Actions for ${TITLE}` }),
  );
  await userEvent.click(
    await screen.findByRole("menuitem", { name: "Rename" }),
  );
  const field = await screen.findByLabelText("Task title");
  await userEvent.clear(field);
  await userEvent.type(field, `${next}{Enter}`);
}

beforeEach(() => {
  reads = 0;
  mockToast.mockClear();
  server.use(
    http.get("/api/proxy/api/tasks", () => {
      reads += 1;
      return HttpResponse.json({
        tasks: [makeTask("task-1", TITLE, "IN_PROGRESS")],
      });
    }),
  );
});

describe("Tasks across tabs", () => {
  it("drops a task another tab archived when a change to it gets a 404", async () => {
    let archivedElsewhere = false;
    server.use(
      http.patch("/api/proxy/api/tasks/task-1", () => {
        archivedElsewhere = true;
        return HttpResponse.json({ detail: "Task not found" }, { status: 404 });
      }),
      http.get("/api/proxy/api/tasks", () =>
        HttpResponse.json({
          tasks: archivedElsewhere
            ? []
            : [makeTask("task-1", TITLE, "IN_PROGRESS")],
        }),
      ),
    );
    render(<Harness />);

    await rename("Plan the retreat");

    await vi.waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "This task no longer exists" }),
      ),
    );
    expect(mockToast).toHaveBeenCalledTimes(1);
    await vi.waitFor(() =>
      expect(screen.queryByRole("button", { name: /^Plan the/ })).toBeNull(),
    );
  });

  it("tells the other tabs about a change, and re-reads when they tell", async () => {
    server.use(
      http.patch("/api/proxy/api/tasks/task-1", () =>
        HttpResponse.json(
          makeTask("task-1", "Plan the retreat", "IN_PROGRESS"),
        ),
      ),
    );
    const heard: unknown[] = [];
    const otherTab = new BroadcastChannel("tasks");
    otherTab.onmessage = (event) => heard.push(event.data);
    render(<Harness />);

    await rename("Plan the retreat");
    await vi.waitFor(() => expect(heard).toContainEqual({ taskID: "task-1" }));

    const before = reads;
    otherTab.postMessage({ taskID: "task-1" });
    await vi.waitFor(() => expect(reads).toBeGreaterThan(before));
    otherTab.close();
  });

  it("knows a 404 means the task is gone", () => {
    const gone = Object.assign(new Error("Task not found"), { status: 404 });
    const busy = Object.assign(new Error("Busy"), { status: 409 });
    expect(isTaskGone(gone)).toBe(true);
    expect(isTaskGone(busy)).toBe(false);
    expect(isTaskGone(new TypeError("Failed to fetch"))).toBe(false);
  });
});
