import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TaskTab } from "../TaskTab";

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: mockToast,
}));

const NOW = new Date(2026, 10, 15, 10, 0, 0).getTime();
const TASK_URL = "/api/proxy/api/tasks/task-1";

let task: Task;
let calls: { url: string; body: unknown }[];

async function bodyOf(request: Request) {
  const text = await request.text();
  return text ? JSON.parse(text) : null;
}

function scheduled(extra: Partial<Task> = {}): Task {
  return {
    ...makeTask("task-1", "Research AutoGPT", "TODO"),
    start_at: new Date(NOW + 4 * 60_000),
    ...extra,
  };
}

function serveTask() {
  server.use(
    http.get(`${TASK_URL}/record`, () =>
      HttpResponse.json({ task, activity: [], artifacts: [], evidence: [] }),
    ),
    http.post(`${TASK_URL}/run`, async ({ request }) => {
      calls.push({ url: "run", body: await bodyOf(request) });
      task = { ...task, status: "IN_PROGRESS", start_at: null };
      return HttpResponse.json(task);
    }),
    http.post(`${TASK_URL}/cancel`, () => {
      calls.push({ url: "cancel", body: null });
      task = { ...task, status: "CANCELLED", start_at: null };
      return HttpResponse.json(task);
    }),
    http.patch(TASK_URL, async ({ request }) => {
      const body = await bodyOf(request);
      calls.push({ url: "update", body });
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
      experts={[]}
      onOpenArtifact={vi.fn()}
    />,
  );
}

async function openEditor() {
  await userEvent.click(await screen.findByRole("button", { name: "Change" }));
  return screen.getByRole("group", { name: "Change the start time" });
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(NOW);
  calls = [];
  mockToast.mockClear();
  task = scheduled();
  serveTask();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Task tab for a scheduled task", () => {
  it("shows when it starts, why it is held, and Start now starts it", async () => {
    task = scheduled({
      status_text: "Waiting to start: your account needs a plan",
    });
    renderTab();

    expect(await screen.findByText("Starts in 4 min")).toBeDefined();
    expect(
      screen.getByText("Waiting to start: your account needs a plan"),
    ).toBeDefined();
    expect(screen.getByText("Scheduled")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Start now" }));

    expect(await screen.findByRole("button", { name: "Stop" })).toBeDefined();
    expect(calls).toEqual([{ url: "run", body: null }]);
  });

  it("cancels it before it starts", async () => {
    renderTab();

    await userEvent.click(
      await screen.findByRole("button", { name: "Cancel" }),
    );

    expect(
      await screen.findByRole("button", { name: "Try again" }),
    ).toBeDefined();
    expect(calls).toEqual([{ url: "cancel", body: null }]);
  });

  it("moves the start time to a quick choice", async () => {
    renderTab();

    const editor = await openEditor();
    await userEvent.click(
      within(editor).getByRole("button", { name: "Tomorrow 9:00" }),
    );

    const tomorrowNine = new Date(2026, 10, 16, 9, 0);
    expect(await screen.findByText(/^Starts tomorrow /)).toBeDefined();
    expect(calls).toEqual([
      { url: "update", body: { start_at: tomorrowNine.toISOString() } },
    ]);
    expect(screen.queryByRole("group", { name: "Change the start time" })).toBe(
      null,
    );
  });

  it("takes the start time off, leaving a plain to-do", async () => {
    renderTab();

    const editor = await openEditor();
    await userEvent.click(
      within(editor).getByRole("button", { name: "Remove start time" }),
    );

    expect(await screen.findByRole("button", { name: "Start" })).toBeDefined();
    expect(calls).toEqual([{ url: "update", body: { start_at: null } }]);
  });

  it("shows why a new start time was refused", async () => {
    server.use(
      http.patch(TASK_URL, () =>
        HttpResponse.json(
          { detail: "Pick a start time in the future." },
          { status: 422 },
        ),
      ),
    );
    renderTab();

    const editor = await openEditor();
    await userEvent.click(
      within(editor).getByRole("button", { name: "In 5 min" }),
    );

    await vi.waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't change the start time",
          description: "Pick a start time in the future.",
          variant: "destructive",
        }),
      ),
    );
  });
});

describe("Task tab for a task that missed its start", () => {
  it("says so quietly and offers Start now and Cancel", async () => {
    task = scheduled({
      start_at: null,
      missed_start: true,
      status_text: "Missed its start time",
    });
    renderTab();

    expect(await screen.findByRole("status")).toHaveProperty(
      "textContent",
      "Missed its start time",
    );
    expect(screen.getByText("Missed start")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Change" })).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Start now" }));

    expect(await screen.findByRole("button", { name: "Stop" })).toBeDefined();
    expect(calls).toEqual([{ url: "run", body: null }]);
  });
});
